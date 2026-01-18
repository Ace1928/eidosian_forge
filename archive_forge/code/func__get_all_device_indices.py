import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _get_all_device_indices():
    return _get_device_attr(lambda m: list(range(m.device_count())))