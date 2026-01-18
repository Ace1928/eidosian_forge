from collections import defaultdict
from copy import deepcopy
import torch
from typing import Any, Optional, Dict
import pytorch_lightning as pl  # type: ignore[import]
from ._data_sparstity_utils import (
def __create_config_based_on_state(self, pl_module):
    config: Dict = defaultdict()
    if self.data_sparsifier_state_dict is None:
        return config
    for name, _ in pl_module.model.named_parameters():
        valid_name = _get_valid_name(name)
        config[valid_name] = self.data_sparsifier.data_groups[valid_name]
    return config