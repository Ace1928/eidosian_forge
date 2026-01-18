import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
def _handle_spike(self, fabric: 'Fabric', batch_idx: int) -> None:
    self.bad_batches.extend([batch_idx - 1, batch_idx])
    if fabric.global_rank == 0:
        assert self.exclude_batches_path is not None
        os.makedirs(os.path.dirname(self.exclude_batches_path), exist_ok=True)
        with open(self.exclude_batches_path, 'w') as f:
            json.dump(self.bad_batches, f, indent=4)
    raise TrainingSpikeException(batch_idx=batch_idx)