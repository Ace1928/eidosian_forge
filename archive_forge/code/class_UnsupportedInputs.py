import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
class UnsupportedInputs(Exception):
    """Exception to be raised during the construction of a :class:`Pair` in case it doesn't support the inputs."""