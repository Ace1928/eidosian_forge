import io
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from packaging import version
from transformers import TrainingArguments
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
from transformers.training_args import OptimizerNames, default_logdir, logger
from transformers.utils import (
from transformers.utils.generic import strtobool
@dataclass
class ORTTrainingArguments(TrainingArguments):
    """
    Parameters:
        optim (`str` or [`training_args.ORTOptimizerNames`] or [`transformers.training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
            The optimizer to use, including optimizers in Transformers: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor. And optimizers implemented by ONNX Runtime: adamw_ort_fused.
    """
    optim: Optional[str] = field(default='adamw_hf', metadata={'help': 'The optimizer to use.'})
    use_module_with_loss: Optional[bool] = field(default=False, metadata={'help': 'Use ModuleWithLoss Wrapper to compute loss inside the training loop, having this will help save memory for ORTModule Runs.'})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)
        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn('using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `IntervalStrategy` instead', FutureWarning)
            self.evaluation_strategy = self.evaluation_strategy.value
        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f'using `logging_steps` to initialize `eval_steps` to {self.logging_steps}')
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(f'evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or --logging_steps')
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f'logging strategy {self.logging_strategy} requires non-zero --logging_steps')
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f'--logging_steps must be an integer if bigger than 1: {self.logging_steps}')
            self.logging_steps = int(self.logging_steps)
        if self.evaluation_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f'--eval_steps must be an integer if bigger than 1: {self.eval_steps}')
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == IntervalStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f'--save_steps must be an integer if bigger than 1: {self.save_steps}')
            self.save_steps = int(self.save_steps)
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(f'--load_best_model_at_end requires the saving steps to be a multiple of the evaluation steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps {self.save_steps} and eval_steps {self.eval_steps}.')
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(f'--load_best_model_at_end requires the saving steps to be a multiple of the evaluation steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps{self.save_steps} and eval_steps {self.eval_steps}.')
                    LARGE_MULTIPLIER = 1000000
                    if self.save_steps * LARGE_MULTIPLIER % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(f'--load_best_model_at_end requires the saving steps to be a multiple of the evaluation steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}.')
                raise ValueError(f'--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}.')
        safetensors_available = is_safetensors_available()
        if self.save_safetensors and (not safetensors_available):
            raise ValueError(f'--save_safetensors={self.save_safetensors} requires safetensors to be installed!')
        if not self.save_safetensors and safetensors_available:
            logger.info(f'Found safetensors installation, but --save_safetensors={self.save_safetensors}. Safetensors should be a preferred weights saving format due to security and performance reasons. If your model cannot be saved by safetensors please feel free to open an issue at https://github.com/huggingface/safetensors!')
        if (self.load_best_model_at_end or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU) and self.metric_for_best_model is None:
            self.metric_for_best_model = 'loss'
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ['loss', 'eval_loss']
        if self.run_name is None:
            self.run_name = self.output_dir
        if self.framework == 'pt' and is_torch_available():
            if self.fp16_backend and self.fp16_backend != 'auto':
                warnings.warn('`fp16_backend` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `half_precision_backend` instead', FutureWarning)
                self.half_precision_backend = self.fp16_backend
            if self.bf16 or self.bf16_full_eval:
                if self.use_cpu and (not is_torch_bf16_cpu_available()):
                    raise ValueError("Your setup doesn't support bf16/(cpu, tpu, neuroncore). You need torch>=1.10")
                elif not self.use_cpu:
                    if torch.cuda.is_available() and (not is_torch_bf16_gpu_available()):
                        raise ValueError("Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU with cuda>=11.0")
        if self.fp16 and self.bf16:
            raise ValueError('At most one of fp16 and bf16 can be True, but not both')
        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError('At most one of fp16 and bf16 can be True for full eval, but not both')
        if self.bf16:
            if self.half_precision_backend == 'apex':
                raise ValueError(' `--half_precision_backend apex`: GPU bf16 is not supported by apex. Use `--half_precision_backend cuda_amp` instead')
        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.evaluation_strategy == IntervalStrategy.NO:
                raise ValueError('lr_scheduler_type reduce_lr_on_plateau requires an eval strategy')
            if not is_torch_available():
                raise ValueError('lr_scheduler_type reduce_lr_on_plateau requires torch>=0.2.0')
        try:
            self.optim = ORTOptimizerNames(self.optim)
        except ValueError:
            self.optim = OptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn('`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim adafactor` instead', FutureWarning)
            self.optim = OptimizerNames.ADAFACTOR
        if self.optim == OptimizerNames.ADAMW_TORCH_FUSED and is_torch_available():
            if version.parse(version.parse(torch.__version__).base_version) < version.parse('2.0.0'):
                raise ValueError('--optim adamw_torch_fused requires PyTorch 2.0 or higher')
            if version.parse(version.parse(torch.__version__).base_version) == version.parse('2.0.0') and self.fp16:
                raise ValueError('--optim adamw_torch_fused with --fp16 requires PyTorch>2.0')
        if is_torch_available() and self.device.type != 'cuda' and (not (self.device.type == 'xla' and 'GPU_NUM_DEVICES' in os.environ)) and (self.fp16 or self.fp16_full_eval):
            raise ValueError('FP16 Mixed precision training with AMP or APEX (`--fp16`) and FP16 half precision evaluation (`--fp16_full_eval`) can only be used on CUDA devices.')
        if is_torch_available() and self.device.type != 'cuda' and (not (self.device.type == 'xla' and 'GPU_NUM_DEVICES' in os.environ)) and (self.device.type != 'cpu') and (self.bf16 or self.bf16_full_eval):
            raise ValueError('BF16 Mixed precision training with AMP (`--bf16`) and BF16 half precision evaluation (`--bf16_full_eval`) can only be used on CUDA or CPU/TPU/NeuronCore devices.')
        if self.torchdynamo is not None:
            warnings.warn('`torchdynamo` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `torch_compile_backend` instead', FutureWarning)
            self.torch_compile_backend = self.torchdynamo
        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and (not self.torch_compile):
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            self.torch_compile_backend = 'inductor'
        if self.torch_compile:
            prefix = 'ACCELERATE_DYNAMO_'
            os.environ[prefix + 'BACKEND'] = self.torch_compile_backend
            if self.torch_compile_mode is not None:
                os.environ[prefix + 'MODE'] = self.torch_compile_mode
        if self.framework == 'pt' and is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and (not self.fp16) or self.bf16:
                    logger.info("Setting TF32 in CUDA backends to speedup torch compile, you won't see any improvement otherwise.")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning('The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here.')
        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    raise ValueError('--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7')
            elif is_torch_tf32_available():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        if self.half_precision_backend != 'apex':
            mixed_precision_dtype = os.environ.get('ACCELERATE_MIXED_PRECISION', 'no')
            if self.fp16:
                mixed_precision_dtype = 'fp16'
            elif self.bf16:
                mixed_precision_dtype = 'bf16'
            os.environ['ACCELERATE_MIXED_PRECISION'] = mixed_precision_dtype
        if self.report_to is None:
            logger.info('The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).')
            self.report_to = 'all'
        if self.report_to == 'all' or self.report_to == ['all']:
            from transformers.integrations import get_available_reporting_integrations
            self.report_to = get_available_reporting_integrations()
        elif self.report_to == 'none' or self.report_to == ['none']:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError('warmup_ratio must lie in range [0,1]')
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info('Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training')
        if isinstance(self.fsdp, bool):
            self.fsdp = 'full_shard' if self.fsdp else ''
        if isinstance(self.fsdp, str):
            self.fsdp = [FSDPOption(s) for s in self.fsdp.split()]
        if self.fsdp == [FSDPOption.OFFLOAD]:
            raise ValueError('`--fsdp offload` can\'t work on its own. It needs to be added to `--fsdp full_shard` or `--fsdp shard_grad_op`. For example, `--fsdp "full_shard offload"`.')
        elif FSDPOption.FULL_SHARD in self.fsdp and FSDPOption.SHARD_GRAD_OP in self.fsdp:
            raise ValueError('`--fsdp full_shard` is not compatible with `--fsdp shard_grad_op`.')
        if self.fsdp_config is None:
            self.fsdp_config = {}
        if isinstance(self.fsdp_config, str):
            if len(self.fsdp) == 0:
                warnings.warn('`--fsdp_config` is useful only when `--fsdp` is specified.')
            with io.open(self.fsdp_config, 'r', encoding='utf-8') as f:
                self.fsdp_config = json.load(f)
                for k in list(self.fsdp_config.keys()):
                    if k.startswith('fsdp_'):
                        v = self.fsdp_config.pop(k)
                        self.fsdp_config[k[5:]] = v
        if self.fsdp_min_num_params > 0:
            warnings.warn('using `--fsdp_min_num_params` is deprecated. Use fsdp_config instead ', FutureWarning)
        self.fsdp_config['min_num_params'] = max(self.fsdp_config.get('min_num_params', 0), self.fsdp_min_num_params)
        if isinstance(self.fsdp_config.get('transformer_layer_cls_to_wrap', None), str):
            self.fsdp_config['transformer_layer_cls_to_wrap'] = [self.fsdp_config['transformer_layer_cls_to_wrap']]
        if self.fsdp_transformer_layer_cls_to_wrap is not None:
            warnings.warn('using `--fsdp_transformer_layer_cls_to_wrap` is deprecated. Use fsdp_config instead ', FutureWarning)
            self.fsdp_config['transformer_layer_cls_to_wrap'] = self.fsdp_config.get('transformer_layer_cls_to_wrap', []) + [self.fsdp_transformer_layer_cls_to_wrap]
        if len(self.fsdp) == 0 and self.fsdp_config['min_num_params'] > 0:
            warnings.warn('`min_num_params` is useful only when `--fsdp` is specified.')
        if len(self.fsdp) == 0 and self.fsdp_config.get('transformer_layer_cls_to_wrap', None) is not None:
            warnings.warn('`transformer_layer_cls_to_wrap` is useful only when `--fsdp` is specified.')
        if len(self.fsdp) > 0 and self.fsdp_config['min_num_params'] > 0 and (self.fsdp_config.get('transformer_layer_cls_to_wrap', None) is not None):
            raise ValueError('`min_num_params` and `transformer_layer_cls_to_wrap` are mutually exclusive.')
        self.fsdp_config['xla'] = self.fsdp_config.get('xla', False)
        self.fsdp_config['xla_fsdp_grad_ckpt'] = self.fsdp_config.get('xla_fsdp_grad_ckpt', False)
        if self.fsdp_config['xla']:
            if len(self.fsdp) > 0:
                self.xla_fsdp_config = self.fsdp_config.get('xla_fsdp_settings', {})
                if 'compute_dtype' in self.xla_fsdp_config:
                    self.xla_fsdp_config['compute_dtype'] = getattr(torch, self.xla_fsdp_config['compute_dtype'])
                if 'buffer_dtype' in self.xla_fsdp_config:
                    self.xla_fsdp_config['buffer_dtype'] = getattr(torch, self.xla_fsdp_config['buffer_dtype'])
            else:
                warnings.warn('XLA FSDP can be used only when `--fsdp` is specified.')
        elif self.fsdp_config['xla_fsdp_grad_ckpt']:
            warnings.warn('`--xla_fsdp_grad_ckpt` is useful only when `--xla` is set to true.')
        if len(self.fsdp) > 0 and (not self.fsdp_config['xla']):
            os.environ['ACCELERATE_USE_FSDP'] = 'true'
            from accelerate.utils.constants import FSDP_AUTO_WRAP_POLICY, FSDP_SHARDING_STRATEGY
            prefix = 'FSDP_'
            for fsdp_option in self.fsdp:
                if fsdp_option.upper() in FSDP_SHARDING_STRATEGY:
                    os.environ[f'{prefix}SHARDING_STRATEGY'] = str(FSDP_SHARDING_STRATEGY.index(fsdp_option.upper()) + 1)
                elif fsdp_option == FSDPOption.OFFLOAD:
                    os.environ[f'{prefix}OFFLOAD_PARAMS'] = 'true'
                elif fsdp_option == FSDPOption.AUTO_WRAP:
                    os.environ[f'{prefix}AUTO_WRAP_POLICY'] = FSDP_AUTO_WRAP_POLICY[0]
                    if self.fsdp_config['min_num_params'] > 0:
                        os.environ[f'{prefix}MIN_NUM_PARAMS'] = str(self.fsdp_config['min_num_params'])
                        os.environ[f'{prefix}AUTO_WRAP_POLICY'] = FSDP_AUTO_WRAP_POLICY[1]
                    elif self.fsdp_config.get('transformer_layer_cls_to_wrap', None) is not None:
                        os.environ[f'{prefix}TRANSFORMER_CLS_TO_WRAP'] = ','.join(self.fsdp_config['transformer_layer_cls_to_wrap'])
            prefetch_policy = self.fsdp_config.get('fsdp_backward_prefetch', 'NO_PREFETCH')
            os.environ[f'{prefix}BACKWARD_PREFETCH'] = prefetch_policy.upper()
            os.environ[f'{prefix}FORWARD_PREFETCH'] = self.fsdp_config.get('forward_prefect', 'false')
            os.environ[f'{prefix}SYNC_MODULE_STATES'] = self.fsdp_config.get('sync_module_states', 'true')
            os.environ[f'{prefix}USE_ORIG_PARAMS'] = self.fsdp_config.get('use_orig_params', 'false')
        if self.tpu_metrics_debug:
            warnings.warn('using `--tpu_metrics_debug` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--debug tpu_metrics_debug` instead', FutureWarning)
            if self.debug is None:
                self.debug = ' tpu_metrics_debug'
            else:
                self.debug += ' tpu_metrics_debug'
            self.tpu_metrics_debug = False
        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]
        elif self.debug is None:
            self.debug = []
        self.deepspeed_plugin = None
        if self.deepspeed:
            if not is_accelerate_available():
                raise ValueError('--deepspeed requires Accelerate to be installed: `pip install accelerate`.')
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)
            from accelerate.utils import DeepSpeedPlugin
            os.environ['ACCELERATE_USE_DEEPSPEED'] = 'true'
            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.hf_deepspeed_config)
        elif strtobool(os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false')):
            from accelerate.utils import DeepSpeedPlugin
            self.deepspeed_plugin = DeepSpeedPlugin()
            mixed_precision = os.environ.get('ACCELERATE_MIXED_PRECISION', 'no')
            self.deepspeed_plugin.set_mixed_precision(mixed_precision)
            self.deepspeed_plugin.set_deepspeed_weakref()
        if self.push_to_hub_token is not None:
            warnings.warn('`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_token` instead.', FutureWarning)
            self.hub_token = self.push_to_hub_token
        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token)
            if self.push_to_hub_organization is not None:
                warnings.warn(f'`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this argument (in this case {self.hub_model_id}).', FutureWarning)
            else:
                warnings.warn(f'`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this argument (in this case {self.hub_model_id}).', FutureWarning)
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f'{self.push_to_hub_organization}/{Path(self.output_dir).name}'
            warnings.warn(f'`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this argument (in this case {self.hub_model_id}).', FutureWarning)
        if self.half_precision_backend != 'apex':
            mixed_precision_dtype = os.environ.get('ACCELERATE_MIXED_PRECISION', 'no')
            if self.fp16:
                mixed_precision_dtype = 'fp16'
            elif self.bf16:
                mixed_precision_dtype = 'bf16'
            os.environ['ACCELERATE_MIXED_PRECISION'] = mixed_precision_dtype
        if self.use_module_with_loss is True:
            logger.info('Using ModuleWithLoss Wrapper.loss will be computed during training loop and it will save memory peak ')
        else:
            logger.info('Not Using ModuleWithLoss Wrapper.')