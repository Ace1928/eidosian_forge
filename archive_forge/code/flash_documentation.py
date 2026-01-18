import os
from dataclasses import replace
from itertools import zip_longest
from typing import Any, List, Optional, Set, Tuple, Union
import torch
from ..common import _get_storage_base, get_operator, register_operator
from .attn_bias import (
from .common import (
Operator that computes memory-efficient attention using         `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_         implementation.
    