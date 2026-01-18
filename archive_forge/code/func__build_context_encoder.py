import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def _build_context_encoder(self):
    """
        Build the context (i.e. dialogue history) encoder.
        """
    if self.opt.get('share_encoder'):
        self.context_encoder = self.label_encoder
    else:
        if self.opt['load_context_encoder_from'] is None and self.opt['context_encoder_embedding_type'] == 'fasttext_cc':
            embeddings = load_fasttext_embeddings(self.dictionary, self.opt['embedding_size'], self.opt['datapath'])
        else:
            embeddings = nn.Embedding(len(self.dictionary), self.opt['embedding_size'])
        self.context_encoder = TransformerEncoder(n_heads=self.opt['n_heads'], n_layers=self.opt['n_layers'], embedding_size=self.opt['embedding_size'], ffn_size=self.opt['ffn_size'], vocabulary_size=len(self.dictionary), embedding=embeddings, dropout=self.opt['dropout'], attention_dropout=self.opt['attention_dropout'], relu_dropout=self.opt['relu_dropout'], padding_idx=self.dictionary.tok2ind[self.dictionary.null_token], learn_positional_embeddings=self.opt['learn_positional_embeddings'], embeddings_scale=False, n_positions=self.opt['n_positions'], activation=self.opt['activation'], variant=self.opt['variant'], n_segments=self.opt['n_segments'])
        if self.opt.get('load_context_encoder_from') is not None:
            self._load_context_encoder_state()