import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
def __uuid__(self):
    """The unique id representing this template"""
    if self._uuid == '':
        self._uuid = to_uuid(self._units, self._template)
    return self._uuid