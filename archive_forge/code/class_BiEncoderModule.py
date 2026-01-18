from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.torch import padded_3d
from parlai.zoo.bert.build import download
from .bert_dictionary import BertDictionaryAgent
from .helpers import (
import os
import torch
from tqdm import tqdm
class BiEncoderModule(torch.nn.Module):
    """
    Groups context_encoder and cand_encoder together.
    """

    def __init__(self, opt):
        super(BiEncoderModule, self).__init__()
        self.context_encoder = BertWrapper(BertModel.from_pretrained(opt['pretrained_path']), opt['out_dim'], add_transformer_layer=opt['add_transformer_layer'], layer_pulled=opt['pull_from_layer'], aggregation=opt['bert_aggregation'])
        self.cand_encoder = BertWrapper(BertModel.from_pretrained(opt['pretrained_path']), opt['out_dim'], add_transformer_layer=opt['add_transformer_layer'], layer_pulled=opt['pull_from_layer'], aggregation=opt['bert_aggregation'])

    def forward(self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt, token_idx_cands, segment_idx_cands, mask_cands):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(token_idx_cands, segment_idx_cands, mask_cands)
        return (embedding_ctxt, embedding_cands)