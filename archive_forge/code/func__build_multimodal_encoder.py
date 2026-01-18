import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def _build_multimodal_encoder(self, n_layers_mm):
    """
        Build the multimodal encoder.

        :param n_layers_mm:
            number of layers for the transformer
        """
    self.multimodal = self.opt.get('multimodal')
    if self.multimodal:
        self.multimodal_combo = self.opt.get('multimodal_combo', 'sum')
        nlayers_mm = self.opt['num_layers_all'] if self.opt['num_layers_all'] != -1 else self.opt['num_layers_multimodal_encoder']
        self.multimodal_encoder = MultimodalCombiner(n_heads=self.opt['n_heads'], n_layers=nlayers_mm, hidden_dim=self.opt['hidden_dim'], ffn_size=self.opt['embedding_size'] * 4, attention_dropout=self.opt['attention_dropout'], relu_dropout=self.opt['relu_dropout'], learn_positional_embeddings=self.opt.get('learn_positional_embeddings', False), reduction=True)