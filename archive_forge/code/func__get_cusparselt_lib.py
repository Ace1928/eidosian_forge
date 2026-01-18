import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def _get_cusparselt_lib() -> Optional[str]:
    libs = glob.glob(str(Path(torch._C.__file__).parent / 'lib' / 'libcusparseLt*.so.0'))
    if len(libs) != 1:
        return None
    return libs[0]