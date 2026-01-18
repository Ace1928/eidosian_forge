import math
from typing import List, Optional, Tuple
import torch
def _apply_pre_attention_layer_norm(self, utterance: torch.Tensor, right_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_norm_input = self.layer_norm_input(torch.cat([right_context, utterance]))
    return (layer_norm_input[right_context.size(0):], layer_norm_input[:right_context.size(0)])