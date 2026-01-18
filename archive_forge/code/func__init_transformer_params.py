import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
def _init_transformer_params(module):
    """
    Initialize the weights of Transformer module in Wav2Vec2/HuBERT.

    If the module is ``nn.Linear``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``bias`` is set to ``True`` in the module, set ``bias`` to 0.

    If the module is ``nn.Embedding``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``padding_idx`` is not None, set the weight of padding to 0.

    Note:
        Ths method corresponds to
        `init_bert_params
        <https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_sentence_encoder.py#L21>`__
        in the original ``fairseq`` implementation.
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()