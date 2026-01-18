import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
def _has_non_default_generation_parameters(self) -> bool:
    """
        Whether or not this instance holds non-default generation parameters.
        """
    for parameter_name, default_value in self._get_generation_defaults().items():
        if hasattr(self, parameter_name) and getattr(self, parameter_name) != default_value:
            return True
    return False