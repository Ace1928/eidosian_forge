from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion

    Fuse the Attention layers into a target class using autoawq

    Args:
        model (`~PreTrainedModel`):
            The input pretrained model
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        modules_to_fuse (`List[str]`):
            The module fusing mapping. The dictionary has to contain a field `attention` with attention module names
            in the correct order: q, k, v, o layer
        current_module_name (`str`):
            The current submodule name
        target_cls (`~autoawq.QuantAttentionFused`):
            The `QuantAttentionFused` class as it only supports that class
            for now.
    