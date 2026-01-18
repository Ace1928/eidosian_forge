from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
class ModelForSCDual(ModelTrunk):

    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        self.seq_classifer = SCHeadDual(self.config_model, dim_embedding=self.config_model['common']['dim_model'], dim_mlp=self.dim_mlp)

    def forward(self, input_ids_0: torch.Tensor, input_ids_1: torch.Tensor, mask_0: torch.Tensor, mask_1: torch.Tensor, label: torch.Tensor):
        mask_0, mask_1 = (mask_0.long(), mask_1.long())
        if self.pooling_mode == Pooling.CLS:
            input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
            input_ids_1, mask_1 = append_cls(input_ids_1, mask_1, self.vocab_size)
        input_ids = torch.cat([input_ids_0, input_ids_1], dim=0)
        masks = torch.cat([mask_0, mask_1], dim=0)
        tokens_out = self.norm(self.model(input_ids, encoder_input_mask=masks)) * masks.unsqueeze(-1)
        seq_scores = self.seq_classifer(*torch.chunk(tokens_out, 2, dim=0))
        seq_loss = torch.nn.CrossEntropyLoss(reduction='none')(seq_scores, label)
        seq_accu = (seq_scores.argmax(dim=-1) == label).to(torch.float32)
        outputs = {'loss': seq_loss.mean(), 'accu': seq_accu.mean(), 'count': label.size(0)}
        return outputs