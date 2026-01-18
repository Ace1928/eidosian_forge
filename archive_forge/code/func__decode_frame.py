import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
def _decode_frame(self, codes: torch.Tensor, scale: Optional[torch.Tensor]=None) -> torch.Tensor:
    codes = codes.transpose(0, 1)
    embeddings = self.quantizer.decode(codes)
    outputs = self.decoder(embeddings)
    if scale is not None:
        outputs = outputs * scale.view(-1, 1, 1)
    return outputs