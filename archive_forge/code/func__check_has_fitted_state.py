import abc
import base64
import collections
import pickle
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ray.air.util.data_batch_conversion import BatchFormat
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
def _check_has_fitted_state(self):
    """Checks if the Preprocessor has fitted state.

        This is also used as an indiciation if the Preprocessor has been fit, following
        convention from Ray versions prior to 2.6.
        This allows preprocessors that have been fit in older versions of Ray to be
        used to transform data in newer versions.
        """
    fitted_vars = [v for v in vars(self) if v.endswith('_')]
    return bool(fitted_vars)