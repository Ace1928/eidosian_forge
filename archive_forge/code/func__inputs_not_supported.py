import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
@staticmethod
def _inputs_not_supported() -> NoReturn:
    raise UnsupportedInputs()