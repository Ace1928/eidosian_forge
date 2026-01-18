from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Parameter
from typing_extensions import override
from pytorch_lightning.utilities.model_summary.model_summary import (
class DeepSpeedLayerSummary(LayerSummary):

    @property
    @override
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum((deepspeed_param_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters()))

    @property
    def average_shard_parameters(self) -> int:
        """Returns the number of parameters in this module."""

        def partitioned_size(p: Parameter) -> int:
            return p.partitioned_size() if RequirementCache('deepspeed<0.6.6') else p.partition_numel()
        return sum((partitioned_size(p) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters()))