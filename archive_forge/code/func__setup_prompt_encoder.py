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
def _setup_prompt_encoder(self, adapter_name: str):
    config = self.peft_config[adapter_name]
    if not hasattr(self, 'prompt_encoder'):
        self.prompt_encoder = torch.nn.ModuleDict({})
        self.prompt_tokens = {}
    transformer_backbone = None
    for name, module in self.base_model.named_children():
        for param in module.parameters():
            param.requires_grad = False
        if isinstance(module, PreTrainedModel):
            if transformer_backbone is None:
                transformer_backbone = module
                self.transformer_backbone_name = name
    if transformer_backbone is None:
        transformer_backbone = self.base_model
    if config.num_transformer_submodules is None:
        config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1
    for named_param, value in list(transformer_backbone.named_parameters()):
        deepspeed_distributed_tensor_shape = getattr(value, 'ds_shape', None)
        if value.shape[0] == self.base_model.config.vocab_size or (deepspeed_distributed_tensor_shape is not None and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size):
            self.word_embeddings = transformer_backbone.get_submodule(named_param.replace('.weight', ''))
            break
    if config.peft_type == PeftType.PROMPT_TUNING:
        prompt_encoder = PromptEmbedding(config, self.word_embeddings)
    elif config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
        prompt_encoder = MultitaskPromptEmbedding(config, self.word_embeddings)
    elif config.peft_type == PeftType.P_TUNING:
        prompt_encoder = PromptEncoder(config)
    elif config.peft_type == PeftType.PREFIX_TUNING:
        prompt_encoder = PrefixEncoder(config)
    else:
        raise ValueError('Not supported')
    prompt_encoder = prompt_encoder.to(self.device)
    self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
    self.prompt_tokens[adapter_name] = torch.arange(config.num_virtual_tokens * config.num_transformer_submodules).long()