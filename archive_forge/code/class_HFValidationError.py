import inspect
import re
import warnings
from functools import wraps
from itertools import chain
from typing import Any, Dict
from ._typing import CallableT
class HFValidationError(ValueError):
    """Generic exception thrown by `huggingface_hub` validators.

    Inherits from [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError).
    """