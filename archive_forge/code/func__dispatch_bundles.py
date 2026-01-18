import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _dispatch_bundles(self, dispatch_all: bool=False) -> None:
    while self._buffer and (dispatch_all or len(self._buffer) >= self._min_buffer_size):
        target_index = self._select_output_index()
        target_bundle = self._pop_bundle_to_dispatch(target_index)
        if self._can_safely_dispatch(target_index, target_bundle.num_rows()):
            target_bundle.output_split_idx = target_index
            self._num_output[target_index] += target_bundle.num_rows()
            self._output_queue.append(target_bundle)
            if self._locality_hints:
                preferred_loc = self._locality_hints[target_index]
                if self._get_location(target_bundle) == preferred_loc:
                    self._locality_hits += 1
                else:
                    self._locality_misses += 1
        else:
            self._buffer.insert(0, target_bundle)
            break