import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _needs_obs_or_fq(prev_output_dtype: Any, prev_output_is_dynamic: bool, cur_target_dtype: Any, cur_target_is_dynamic: bool, reuse_input_obs_or_fq: bool, is_zeroth_arg: bool=False) -> bool:
    """
    note: we will treat "not specified" as torch.float for now
    utility function that checks if we should insert an observer or fake quant node
    base on the requested dtype for the nodes from user

    is_zeroth_arg: we only dynamically quantize the first arg of the node right now
      this should be removed when we enable configuring dynamic quantization
      for a specific argument, this can be removed if we deprecate fx graph mode
      quantization

    """
    if cur_target_is_dynamic:
        assert cur_target_dtype in _OBS_DTYPE_LIST, f'Expected cur_target_dtype to be torch.float, but got: {cur_target_dtype}'
        assert prev_output_dtype not in _DO_NOT_OBS_DTYPE_LIST
        return is_zeroth_arg
    if reuse_input_obs_or_fq:
        return False
    if cur_target_dtype in _OBS_DTYPE_LIST:
        return prev_output_dtype in _OBS_DTYPE_LIST + [torch.float] and cur_target_dtype != prev_output_dtype
    return False