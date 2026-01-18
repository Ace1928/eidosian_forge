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
class _ArrowListWrapper:

    def __init__(self, arrow_col):
        self.arrow_col = arrow_col

    def __getitem__(self, i):
        return self.arrow_col[i].as_py()

    def __len__(self):
        return len(self.arrow_col)