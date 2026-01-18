from typing import Any, Dict, List, Optional
import torch
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier.utils import fqn_to_module, module_to_fqn
import warnings
def __get_state__(self) -> Dict[str, Any]:
    data_groups = self._get_serializable_data_groups()
    state = self._convert_mask(self.state)
    return {'defaults': self.defaults, 'state': state, 'data_groups': data_groups}