import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
def _process_model_before_weight_loading(self, model: 'PreTrainedModel', device_map, keep_in_fp32_modules: List[str]=[], **kwargs):
    from ..integrations import get_keys_to_not_convert, replace_with_bnb_linear
    load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload
    if self.quantization_config.llm_int8_skip_modules is None:
        self.modules_to_not_convert = get_keys_to_not_convert(model)
    else:
        self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules
    if not isinstance(self.modules_to_not_convert, list):
        self.modules_to_not_convert = [self.modules_to_not_convert]
    self.modules_to_not_convert.extend(keep_in_fp32_modules)
    if isinstance(device_map, dict) and len(device_map.keys()) > 1:
        keys_on_cpu = [key for key, value in device_map.items() if value in ['disk', 'cpu']]
        if len(keys_on_cpu) > 0 and (not load_in_8bit_fp32_cpu_offload):
            raise ValueError('If you want to offload some keys to `cpu` or `disk`, you need to set `llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be  converted to 8-bit but kept in 32-bit.')
        self.modules_to_not_convert.extend(keys_on_cpu)
    model = replace_with_bnb_linear(model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config)
    model.config.quantization_config = self.quantization_config