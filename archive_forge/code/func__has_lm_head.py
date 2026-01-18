import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .modeling_base import PreTrainedModelWrapper
def _has_lm_head(self):
    for name, _module in self.pretrained_model.named_modules():
        if any((attribute in name for attribute in self.lm_head_namings)):
            return True
    return False