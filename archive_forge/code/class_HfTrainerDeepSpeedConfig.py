import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):
    """
    The `HfTrainerDeepSpeedConfig` object is meant to be created during `TrainingArguments` object creation and has the
    same lifespan as the latter.
    """

    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = None
        self.mismatches = []

    def dtype(self):
        if self._dtype is None:
            raise ValueError("trainer_config_process() wasn't called yet to tell dtype")
        return self._dtype

    def is_auto(self, ds_key_long):
        val = self.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == 'auto'

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):
        """
        A utility method that massages the config file and can optionally verify that the values match.

        1. Replace "auto" values with `TrainingArguments` value.

        2. If it wasn't "auto" and `must_match` is true, then check that DS config matches Trainer
        config values and if mismatched add the entry to `self.mismatched` - will assert during
        `trainer_config_finalize` for one or more mismatches.

        """
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return
        if config.get(ds_key) == 'auto':
            config[ds_key] = hf_val
            return
        if not must_match:
            return
        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f'- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}')
    fill_only = partialmethod(fill_match, must_match=False)

    def trainer_config_process(self, args, auto_find_batch_size=False):
        """
        Adjust the config with `TrainingArguments` values. This stage is run during `TrainingArguments` object
        creation.
        """
        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match('train_micro_batch_size_per_gpu', args.per_device_train_batch_size, 'per_device_train_batch_size', not auto_find_batch_size)
        self.fill_match('gradient_accumulation_steps', args.gradient_accumulation_steps, 'gradient_accumulation_steps')
        self.fill_match('train_batch_size', train_batch_size, 'train_batch_size (calculated)', not auto_find_batch_size)
        self.fill_match('gradient_clipping', args.max_grad_norm, 'max_grad_norm')
        self.fill_match('optimizer.params.lr', args.learning_rate, 'learning_rate')
        self.fill_match('optimizer.params.betas', [args.adam_beta1, args.adam_beta2], 'adam_beta1+adam_beta2')
        self.fill_match('optimizer.params.eps', args.adam_epsilon, 'adam_epsilon')
        self.fill_match('optimizer.params.weight_decay', args.weight_decay, 'weight_decay')
        self.fill_only('scheduler.params.warmup_min_lr', 0)
        self.fill_match('scheduler.params.warmup_max_lr', args.learning_rate, 'learning_rate')
        if args.fp16 or args.fp16_full_eval:
            fp16_backend = 'apex' if args.fp16_backend == 'apex' else 'amp'
        else:
            fp16_backend = None
        if args.save_on_each_node:
            self.config['checkpoint'] = self.config.get('checkpoint', {})
            self.config['checkpoint']['use_node_local_storage'] = args.save_on_each_node
        self.fill_match('fp16.enabled', (args.fp16 or args.fp16_full_eval) and fp16_backend == 'amp', 'fp16|fp16_full_eval+fp16_backend(amp)')
        self.fill_match('amp.enabled', fp16_backend == 'apex', 'fp16+fp16_backend(apex)')
        self.fill_match('amp.opt_level', args.fp16_opt_level, 'fp16_opt_level')
        self.fill_match('bf16.enabled', args.bf16 or args.bf16_full_eval, 'bf16|bf16_full_eval')
        if self.is_true('bf16.enabled'):
            self._dtype = torch.bfloat16
        elif self.is_false('fp16.enabled'):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16

    def trainer_config_finalize(self, args, model, num_training_steps):
        """
        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        """
        hidden_size_based_keys = ['zero_optimization.reduce_bucket_size', 'zero_optimization.stage3_prefetch_bucket_size', 'zero_optimization.stage3_param_persistence_threshold']
        hidden_size_auto_keys = [x for x in hidden_size_based_keys if self.is_auto(x)]
        if len(hidden_size_auto_keys) > 0:
            if hasattr(model.config, 'hidden_size'):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, 'hidden_sizes'):
                hidden_size = max(model.config.hidden_sizes)
            else:
                raise ValueError(f"The model's config file has neither `hidden_size` nor `hidden_sizes` entry, therefore it's not possible to automatically fill out the following `auto` entries in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing `auto` values for these keys with an integer value of your choice.")
            self.fill_only('zero_optimization.reduce_bucket_size', hidden_size * hidden_size)
            if self.is_zero3():
                self.fill_only('zero_optimization.stage3_prefetch_bucket_size', 0.9 * hidden_size * hidden_size)
                self.fill_only('zero_optimization.stage3_param_persistence_threshold', 10 * hidden_size)
        self.fill_match('scheduler.params.total_num_steps', num_training_steps, 'num_training_steps (calculated)')
        self.fill_match('scheduler.params.warmup_num_steps', args.get_warmup_steps(num_training_steps), 'warmup_steps')
        if len(self.mismatches) > 0:
            mismatches = '\n'.join(self.mismatches)
            raise ValueError(f"Please correct the following DeepSpeed config values that mismatch TrainingArguments values:\n{mismatches}\nThe easiest method is to set these DeepSpeed config values to 'auto'.")