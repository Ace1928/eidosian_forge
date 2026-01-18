import abc
import base64
import collections
import pickle
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ray.air.util.data_batch_conversion import BatchFormat
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
@PublicAPI(stability='beta')
class PreprocessorNotFittedException(RuntimeError):
    """Error raised when the preprocessor needs to be fitted first."""
    pass