import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (`int`): Index of the first character in the original string.
        end (`int`): Index of the character following the last character in the original string.
    """
    start: int
    end: int