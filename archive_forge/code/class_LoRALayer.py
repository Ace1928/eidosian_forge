import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
class LoRALayer:

    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights