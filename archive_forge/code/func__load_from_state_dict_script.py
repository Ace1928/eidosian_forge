import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _load_from_state_dict_script(self, state_dict: Dict[str, Any], prefix: str, local_metadata: Dict[str, torch.Tensor], strict: bool, missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
    self._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)