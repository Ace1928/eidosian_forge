from __future__ import annotations
import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union
import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from . import __version__
from .config import PeftConfig
from .tuners import (
from .utils import (
def _prefix_tuning_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
    batch_size = _get_batch_size(input_ids, inputs_embeds)
    past_key_values = self.get_prompt(batch_size)
    fwd_params = list(inspect.signature(self.base_model.forward).parameters.keys())
    kwargs.update({'input_ids': input_ids, 'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds, 'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states, 'return_dict': return_dict, 'past_key_values': past_key_values})
    if 'past_key_values' in fwd_params:
        return self.base_model(start_positions=start_positions, end_positions=end_positions, **kwargs)
    else:
        transformer_backbone_name = self.base_model.get_submodule(self.transformer_backbone_name)
        fwd_params = list(inspect.signature(transformer_backbone_name.forward).parameters.keys())
        if 'past_key_values' not in fwd_params:
            raise ValueError('Model does not support past key values which are required for prefix tuning.')
        outputs = transformer_backbone_name(**kwargs)
        sequence_output = outputs[0]
        if 'dropout' in [name for name, _ in list(self.base_model.named_children())]:
            sequence_output = self.base_model.dropout(sequence_output)
        logits = self.base_model.get_submodule(self.cls_layer_name)(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return (total_loss,) + output if total_loss is not None else output
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)