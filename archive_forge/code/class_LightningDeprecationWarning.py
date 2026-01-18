import logging
import os
from functools import wraps
from typing import Callable, Optional, TypeVar, overload
import lightning_utilities.core.rank_zero as rank_zero_module
from lightning_utilities.core.rank_zero import (  # noqa: F401
from typing_extensions import ParamSpec
from lightning_fabric.utilities.imports import _UTILITIES_GREATER_EQUAL_0_10
class LightningDeprecationWarning(DeprecationWarning):
    """Deprecation warnings raised by Lightning."""