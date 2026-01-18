import math
from collections import deque
from typing import Any, Dict, List, Optional
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.util import locality_string
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef
def _pop_bundle_to_dispatch(self, target_index: int) -> RefBundle:
    if self._locality_hints:
        preferred_loc = self._locality_hints[target_index]
        for bundle in self._buffer:
            if self._get_location(bundle) == preferred_loc:
                self._buffer.remove(bundle)
                return bundle
    return self._buffer.pop(0)