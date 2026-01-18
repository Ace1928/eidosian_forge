from enum import Enum
from functools import reduce
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from parlai.agents.transformer.modules import (
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
def _forward_early_fusion(self, src_tokens: Optional[torch.LongTensor], image_features: Optional[Union[List[object], torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
        Encode images with context.

        Performs early fusion, whereby image embeddings and token embeddings are computed
        before passing into the Transformer.

        Essentially overrides normal TransformerEncoder forward.
        """
    context_tensor = context_mask = None
    image_tensor = image_mask = None
    if src_tokens is not None and image_features is not None:
        assert src_tokens.size(0) == len(image_features)
    if src_tokens is not None:
        context_tensor, context_mask = self.forward_embedding(src_tokens, segments=torch.zeros_like(src_tokens))
    if image_features is not None:
        valid_imgs = [v for v in image_features if isinstance(v, torch.Tensor)]
        segments: Optional[torch.LongTensor] = None
        if valid_imgs:
            segments = torch.ones((len(image_features), self.n_image_channels * self.n_image_tokens), dtype=torch.long, device=valid_imgs[0].device)
        image_tensor, image_mask = self.encode_images(image_features, segments=segments)
    tensor = self._cat([context_tensor, image_tensor])
    mask: torch.BoolTensor = self._cat([context_mask, image_mask])
    if self.variant == 'xlm':
        tensor = _normalize(tensor, self.norm_embeddings)
    tensor = self.dropout(tensor)
    tensor *= mask.unsqueeze(-1).type_as(tensor)
    tensor = self.forward_layers(tensor, mask)
    if self.variant == 'prelayernorm':
        tensor = _normalize(tensor, self.norm_embeddings)
    tensor, out_mask = self.reduce_output(tensor, mask)
    return (tensor, out_mask)