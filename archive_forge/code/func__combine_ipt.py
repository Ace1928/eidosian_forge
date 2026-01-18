import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def _combine_ipt(self, ipt_E, ipt_AB):
    ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
    sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
    return sum_ipt