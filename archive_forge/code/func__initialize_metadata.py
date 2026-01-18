from typing import Callable, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.stats import StatsDict
def _initialize_metadata(self):
    assert self._input_data is not None and self._is_input_initialized
    self._num_output_bundles = len(self._input_data)
    block_metadata = []
    for bundle in self._input_data:
        block_metadata.extend([m for _, m in bundle.blocks])
    self._stats = {'input': block_metadata}