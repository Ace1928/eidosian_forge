import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t_v2 import SeamlessM4Tv2Config
@dataclass
class SeamlessM4Tv2GenerationOutput(ModelOutput):
    """
    Class defining the generated outputs from [`SeamlessM4Tv2Model`], [`SeamlessM4Tv2ForTextToText`],
    [`SeamlessM4Tv2ForTextToSpeech`], [`SeamlessM4Tv2ForSpeechToSpeech`] and [`SeamlessM4Tv2ForTextToSpeech`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
        waveform_lengths (`torch.IntTensor` of shape `(batch_size,)`, *optional*):
            The length in samples of each element in the `waveform` batch.
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The generated translated sequences. This is the output of the text-to-text or the speech-to-text models.
            The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches finished
            early due to the `eos_token_id`.
        unit_sequences (`torch.LongTensor` of shape `(batch_size, unit_sequence_length)`, *optional*):
            The generated translated unit sequences. This is the output of the text-to-units model. The second
            dimension (unit_sequence_length) is either equal to `t2u_max_length` or shorter if all batches finished
            early due to the `t2u_eos_token_id`.
    """
    waveform: Optional[torch.FloatTensor] = None
    waveform_lengths: Optional[torch.IntTensor] = None
    sequences: Optional[Tuple[torch.FloatTensor]] = None
    unit_sequences: Optional[Tuple[torch.FloatTensor]] = None