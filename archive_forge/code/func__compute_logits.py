import logging
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from .wavlm_attention import WavLMSelfAttention
def _compute_logits(proj_x: Tensor, target: Tensor, label_embeddings: Parameter) -> Tensor:
    """Compute the logits of the embeddings.
    Args:
        proj_x (Tensor): The projected masked representations of dimension `[batch, frame, final_dim]`.
        target (Tensor): The target Tensor of dimension `[batch, frame, final_dim]`.
        label_embeddings (Parameter): The trainable embeddings of target of dimension `[num_class, final_dim]`.

    Returns:
        (Tensor): The logits of the inputs.
    """
    logit_temp = 0.1
    pos = torch.index_select(label_embeddings, 0, target.long())
    negs = label_embeddings.unsqueeze(1).expand(-1, proj_x.size(0), -1)
    neg_is_pos = (pos == negs).all(-1)
    pos = pos.unsqueeze(0)
    targets = torch.cat([pos, negs], dim=0)
    logits = torch.cosine_similarity(proj_x.float(), targets.float(), dim=-1).type_as(proj_x)
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float('-inf')
    logits = logits.transpose(0, 1)
    return logits