import logging
import numpy as np
import os
import sys
from typing import Any, Optional
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.typing import TensorShape, TensorType
class _TFStub:

    def __init__(self) -> None:
        self.keras = _KerasStub()

    def __bool__(self):
        return False