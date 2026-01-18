import math
import torch
import torch.nn as nn
from fairscale.nn.moe.moe_layer import MOELayer
from fairscale.nn.moe.top2gate import Top2Gate
class TransformerLM(nn.Sequential):
    """A GPT-2 based nn.Sequential language model."""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder, is_moe=False, num_local_experts=1):
        layers = [EmbeddingLayer(ntokens, ninp, initrange), PositionalEncodingLayer(ninp, dropout)]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout, is_moe, num_local_experts))
        layers.append(LinearLayer(ninp, ntokens, initrange))
        super(TransformerLM, self).__init__(*layers)