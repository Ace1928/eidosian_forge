import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def _build_text_encoder(self, n_layers_text):
    """
        Build the text (candidate) encoder.

        :param n_layers_text:
            how many layers the transformer will have
        """
    self.embeddings = nn.Embedding(len(self.dictionary), self.opt['embedding_size'])
    if self.opt.get('load_encoder_from') is None and self.opt['embedding_type'] == 'fasttext_cc':
        self.embeddings = load_fasttext_embeddings(self.dictionary, self.opt['embedding_size'], self.opt['datapath'])
    self.text_encoder = TransformerEncoder(n_heads=self.opt['n_heads'], n_layers=self.opt['n_layers'], embedding_size=self.opt['embedding_size'], ffn_size=self.opt['ffn_size'], vocabulary_size=len(self.dictionary), embedding=self.embeddings, dropout=self.opt['dropout'], attention_dropout=self.opt['attention_dropout'], relu_dropout=self.opt['relu_dropout'], padding_idx=self.dictionary.tok2ind[self.dictionary.null_token], learn_positional_embeddings=self.opt['learn_positional_embeddings'], embeddings_scale=False, n_positions=self.opt['n_positions'], activation=self.opt['activation'], variant=self.opt['variant'], n_segments=self.opt['n_segments'])
    if self.opt.get('load_encoder_from') is not None:
        self._load_text_encoder_state()
    self.additional_layer = LinearWrapper(self.opt['embedding_size'], self.opt['hidden_dim'], dropout=self.opt['additional_layer_dropout'])