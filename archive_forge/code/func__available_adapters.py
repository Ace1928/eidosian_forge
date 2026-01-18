import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
@property
def _available_adapters(self) -> Set[str]:
    return {*self.hada_w1_a, *self.hada_w1_b, *self.hada_w2_a, *self.hada_w2_b, *self.hada_t1, *self.hada_t2}