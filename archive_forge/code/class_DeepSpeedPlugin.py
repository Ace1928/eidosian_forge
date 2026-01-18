import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class DeepSpeedPlugin:
    """
    This plugin is used to integrate DeepSpeed.
    """
    hf_ds_config: Any = field(default=None, metadata={'help': 'path to DeepSpeed config file or dict or an object of class `accelerate.utils.deepspeed.HfDeepSpeedConfig`.'})
    gradient_accumulation_steps: int = field(default=None, metadata={'help': 'Number of steps to accumulate gradients before updating optimizer states. If not set, will use the value from the `Accelerator` directly.'})
    gradient_clipping: float = field(default=None, metadata={'help': 'Enable gradient clipping with value'})
    zero_stage: int = field(default=None, metadata={'help': 'Possible options are 0,1,2,3; Default will be taken from environment variable'})
    is_train_batch_min: str = field(default=True, metadata={'help': 'If both train & eval dataloaders are specified, this will decide the train_batch_size'})
    offload_optimizer_device: bool = field(default=None, metadata={'help': 'Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.'})
    offload_param_device: bool = field(default=None, metadata={'help': 'Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.'})
    offload_optimizer_nvme_path: str = field(default=None, metadata={'help': 'Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.'})
    offload_param_nvme_path: str = field(default=None, metadata={'help': 'Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.'})
    zero3_init_flag: bool = field(default=None, metadata={'help': 'Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models.Only applicable with ZeRO Stage-3.'})
    zero3_save_16bit_model: bool = field(default=None, metadata={'help': 'Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.'})

    def __post_init__(self):
        from .deepspeed import HfDeepSpeedConfig
        if self.gradient_accumulation_steps is None:
            gas = os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', 'auto')
            self.gradient_accumulation_steps = int(gas) if gas.isdigit() else gas
        if self.gradient_clipping is None:
            gradient_clipping = os.environ.get('ACCELERATE_GRADIENT_CLIPPING', 'none')
            if gradient_clipping != 'none':
                self.gradient_clipping = float(gradient_clipping)
        if self.zero_stage is None:
            self.zero_stage = int(os.environ.get('ACCELERATE_DEEPSPEED_ZERO_STAGE', 2))
        if self.offload_optimizer_device is None:
            self.offload_optimizer_device = os.environ.get('ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE', 'none')
        if self.offload_param_device is None:
            self.offload_param_device = os.environ.get('ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE', 'none')
        if self.offload_optimizer_nvme_path is None:
            self.offload_optimizer_nvme_path = os.environ.get('ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH', 'none')
        if self.offload_param_nvme_path is None:
            self.offload_param_nvme_path = os.environ.get('ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH', 'none')
        if self.zero3_save_16bit_model is None:
            self.zero3_save_16bit_model = os.environ.get('ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL', 'false') == 'true'
        if self.hf_ds_config is None:
            self.hf_ds_config = os.environ.get('ACCELERATE_DEEPSPEED_CONFIG_FILE', 'none')
        if isinstance(self.hf_ds_config, dict) or (isinstance(self.hf_ds_config, str) and self.hf_ds_config != 'none') or isinstance(self.hf_ds_config, HfDeepSpeedConfig):
            if not isinstance(self.hf_ds_config, HfDeepSpeedConfig):
                self.hf_ds_config = HfDeepSpeedConfig(self.hf_ds_config)
            if 'gradient_accumulation_steps' not in self.hf_ds_config.config:
                self.hf_ds_config.config['gradient_accumulation_steps'] = 1
            if 'zero_optimization' not in self.hf_ds_config.config:
                raise ValueError('Please specify the ZeRO optimization config in the DeepSpeed config.')
            self._deepspeed_config_checks()
            plugin_to_config_mapping = {'gradient_accumulation_steps': 'gradient_accumulation_steps', 'gradient_clipping': 'gradient_clipping', 'zero_stage': 'zero_optimization.stage', 'offload_optimizer_device': 'zero_optimization.offload_optimizer.device', 'offload_param_device': 'zero_optimization.offload_param.device', 'offload_param_nvme_path': 'zero_optimization.offload_param.nvme_path', 'offload_optimizer_nvme_path': 'zero_optimization.offload_optimizer.nvme_path', 'zero3_save_16bit_model': 'zero_optimization.stage3_gather_16bit_weights_on_model_save'}
            kwargs = {v: getattr(self, k) for k, v in plugin_to_config_mapping.items() if getattr(self, k) is not None}
            for key in kwargs.keys():
                self.fill_match(key, **kwargs, must_match=False)
            self.hf_ds_config.set_stage_and_offload()
            for key, value in plugin_to_config_mapping.items():
                config_value = self.hf_ds_config.get_value(value)
                if config_value is not None and config_value != 'auto':
                    setattr(self, key, config_value)
        else:
            config = {'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': self.gradient_accumulation_steps, 'zero_optimization': {'stage': self.zero_stage, 'offload_optimizer': {'device': self.offload_optimizer_device, 'nvme_path': self.offload_optimizer_nvme_path if self.offload_optimizer_device == 'nvme' else None}, 'offload_param': {'device': self.offload_param_device, 'nvme_path': self.offload_param_nvme_path if self.offload_param_device == 'nvme' else None}, 'stage3_gather_16bit_weights_on_model_save': self.zero3_save_16bit_model}}
            if self.gradient_clipping:
                config['gradient_clipping'] = self.gradient_clipping
            self.hf_ds_config = HfDeepSpeedConfig(config)
        self.deepspeed_config = self.hf_ds_config.config
        self.deepspeed_config['steps_per_print'] = float('inf')
        if self.zero3_init_flag is None:
            self.zero3_init_flag = str_to_bool(os.environ.get('ACCELERATE_DEEPSPEED_ZERO3_INIT', str(self.hf_ds_config.is_zero3()))) == 1
        if self.zero3_init_flag and (not self.hf_ds_config.is_zero3()):
            warnings.warn('DeepSpeed Zero3 Init flag is only applicable for ZeRO Stage 3. Setting it to False.')
            self.zero3_init_flag = False

    def fill_match(self, ds_key_long, mismatches=None, must_match=True, **kwargs):
        mismatches = [] if mismatches is None else mismatches
        config, ds_key = self.hf_ds_config.find_config_node(ds_key_long)
        if config is None:
            return
        if config.get(ds_key) == 'auto':
            if ds_key_long in kwargs:
                config[ds_key] = kwargs[ds_key_long]
                return
            else:
                raise ValueError(f'`{ds_key_long}` not found in kwargs. Please specify `{ds_key_long}` without `auto` (set to correct value) in the DeepSpeed config file or pass it in kwargs.')
        if not must_match:
            return
        ds_val = config.get(ds_key)
        if ds_val is not None and ds_key_long in kwargs:
            if ds_val != kwargs[ds_key_long]:
                mismatches.append(f'- ds {ds_key_long}={ds_val} vs arg {ds_key_long}={kwargs[ds_key_long]}')

    def is_auto(self, ds_key_long):
        val = self.hf_ds_config.get_value(ds_key_long)
        if val is None:
            return False
        else:
            return val == 'auto'

    def get_value(self, ds_key_long, default=None):
        return self.hf_ds_config.get_value(ds_key_long, default)

    def deepspeed_config_process(self, prefix='', mismatches=None, config=None, must_match=True, **kwargs):
        """Process the DeepSpeed config with the values from the kwargs."""
        mismatches = [] if mismatches is None else mismatches
        if config is None:
            config = self.deepspeed_config
        for key, value in config.items():
            if isinstance(value, dict):
                self.deepspeed_config_process(prefix=prefix + key + '.', mismatches=mismatches, config=value, must_match=must_match, **kwargs)
            else:
                self.fill_match(prefix + key, mismatches, must_match=must_match, **kwargs)
        if len(mismatches) > 0 and prefix == '':
            mismatches_msg = '\n'.join(mismatches)
            raise ValueError(f"Please correct the following DeepSpeed config values that mismatch kwargs  values:\n{mismatches_msg}\nThe easiest method is to set these DeepSpeed config values to 'auto'.")

    def set_mixed_precision(self, mixed_precision):
        ds_config = self.deepspeed_config
        kwargs = {'fp16.enabled': mixed_precision == 'fp16', 'bf16.enabled': mixed_precision == 'bf16'}
        if mixed_precision == 'fp16':
            if 'fp16' not in ds_config:
                ds_config['fp16'] = {'enabled': True, 'auto_cast': True}
        elif mixed_precision == 'bf16':
            if 'bf16' not in ds_config:
                ds_config['bf16'] = {'enabled': True}
        if mixed_precision != 'no':
            diff_dtype = 'bf16' if mixed_precision == 'fp16' else 'fp16'
            if str(ds_config.get(diff_dtype, {}).get('enabled', 'False')).lower() == 'true':
                raise ValueError(f'`--mixed_precision` arg cannot be set to `{mixed_precision}` when `{diff_dtype}` is set in the DeepSpeed config file.')
        for dtype in ['fp16', 'bf16']:
            if dtype not in ds_config:
                ds_config[dtype] = {'enabled': False}
        self.fill_match('fp16.enabled', must_match=False, **kwargs)
        self.fill_match('bf16.enabled', must_match=False, **kwargs)

    def set_deepspeed_weakref(self):
        from .imports import is_transformers_available
        if self.zero3_init_flag:
            if not is_transformers_available():
                raise Exception('When `zero3_init_flag` is set, it requires Transformers to be installed. Please run `pip install transformers`.')
            ds_config = copy.deepcopy(self.deepspeed_config)
            if 'gradient_accumulation_steps' not in ds_config or ds_config['gradient_accumulation_steps'] == 'auto':
                ds_config['gradient_accumulation_steps'] = 1
            if 'train_micro_batch_size_per_gpu' not in ds_config or ds_config['train_micro_batch_size_per_gpu'] == 'auto':
                ds_config['train_micro_batch_size_per_gpu'] = 1
            if ds_config.get('train_batch_size', None) == 'auto':
                del ds_config['train_batch_size']
            if compare_versions('transformers', '<', '4.33'):
                from transformers.deepspeed import HfDeepSpeedConfig
            else:
                from transformers.integrations import HfDeepSpeedConfig
            self.dschf = HfDeepSpeedConfig(ds_config)

    def is_zero3_init_enabled(self):
        return self.zero3_init_flag

    @contextmanager
    def zero3_init_context_manager(self, enable=False):
        old = self.zero3_init_flag
        if old == enable:
            yield
        else:
            self.zero3_init_flag = enable
            self.dschf = None
            self.set_deepspeed_weakref()
            yield
            self.zero3_init_flag = old
            self.dschf = None
            self.set_deepspeed_weakref()

    def _deepspeed_config_checks(self):
        env_variable_names_to_ignore = ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS', 'ACCELERATE_GRADIENT_CLIPPING', 'ACCELERATE_DEEPSPEED_ZERO_STAGE', 'ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE', 'ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE', 'ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH', 'ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH', 'ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL', 'ACCELERATE_MIXED_PRECISION']
        env_variable_names_to_ignore = [name.replace('ACCELERATE_', '').replace('DEEPSPEED_', '').lower() for name in env_variable_names_to_ignore]
        deepspeed_fields_from_accelerate_config = os.environ.get('ACCELERATE_CONFIG_DS_FIELDS', '').split(',')
        if any((name in env_variable_names_to_ignore for name in deepspeed_fields_from_accelerate_config)):
            raise ValueError(f'When using `deepspeed_config_file`, the following accelerate config variables will be ignored: {env_variable_names_to_ignore}.\nPlease specify them appropriately in the DeepSpeed config file.\nIf you are using an accelerate config file, remove others config variables mentioned in the above specified list.\nThe easiest method is to create a new config following the questionnaire via `accelerate config`.\nIt will only ask for the necessary config variables when using `deepspeed_config_file`.')