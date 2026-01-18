from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
class ModelForSC(ModelTrunk):

    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        self.seq_classifer = SCHead(self.config_model, dim_embedding=self.config_model['common']['dim_model'], dim_mlp=self.dim_mlp)

    def forward(self, input_ids_0: torch.Tensor, mask_0: torch.Tensor, label: torch.Tensor):
        if self.pooling_mode == Pooling.CLS:
            input_ids_0, mask_0 = append_cls(input_ids_0, mask_0, self.vocab_size)
        token_out = self.norm(self.model(input_ids_0, encoder_input_mask=mask_0)) * mask_0.unsqueeze(-1)
        seq_scores = self.seq_classifer(token_out)
        seq_loss = torch.nn.CrossEntropyLoss(reduction='none')(seq_scores, label)
        seq_accu = (seq_scores.argmax(dim=-1) == label).to(torch.float32)
        outputs = {'loss': seq_loss.mean(), 'accu': seq_accu.mean(), 'count': label.size(0)}
        return outputs