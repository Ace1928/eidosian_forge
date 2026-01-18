import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
class _NNStub:

    def __init__(self, *a, **kw):
        self.functional = None
        self.Module = _FakeTorchClassStub
        self.parallel = _ParallelStub()