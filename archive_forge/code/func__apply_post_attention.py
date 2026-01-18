import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
def _apply_post_attention(self, rc_output: torch.Tensor, ffn0_out: torch.Tensor, conv_cache: Optional[torch.Tensor], rc_length: int, utterance_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = self.dropout(rc_output) + ffn0_out[:rc_length + utterance_length]
    conv_utterance, conv_right_context, conv_cache = self.conv(result[rc_length:], result[:rc_length], conv_cache)
    result = torch.cat([conv_right_context, conv_utterance])
    result = self.ffn1(result)
    result = self.layer_norm_output(result)
    output_utterance, output_right_context = (result[rc_length:], result[:rc_length])
    return (output_utterance, output_right_context, conv_cache)