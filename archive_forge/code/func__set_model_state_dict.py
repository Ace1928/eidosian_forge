import os
import torch
from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, FSDP_PYTORCH_VERSION, OPTIMIZER_NAME
from .imports import is_torch_distributed_available
from .modeling import is_peft_model
from .versions import is_torch_version
def _set_model_state_dict(model, state_dict, adapter_only=False):
    if adapter_only and is_peft_model(model):
        from peft import set_peft_model_state_dict
        return set_peft_model_state_dict(model, state_dict, adapter_name=model.active_adapter)
    else:
        return model.load_state_dict(state_dict)