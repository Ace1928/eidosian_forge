import bisect
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional
import numpy as np
import ray
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockAccessor
from ray.data.context import DataContext
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def _get_bounds(block, key):
    if len(block) == 0:
        return None
    b = (block[key][0], block[key][len(block) - 1])
    if isinstance(block, pa.Table):
        b = (b[0].as_py(), b[1].as_py())
    return b