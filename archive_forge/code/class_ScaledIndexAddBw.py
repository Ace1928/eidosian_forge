from typing import Optional, Sequence
import torch
from xformers.ops._triton import (
from .common import BaseOperator, register_operator
@register_operator
class ScaledIndexAddBw(BaseOperator):
    OPERATOR = scaled_index_add_bwd
    OPERATOR_CATEGORY = 'indexing'
    NAME = 'scaled_index_addB'