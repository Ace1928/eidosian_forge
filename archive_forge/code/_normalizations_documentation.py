from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
Normalize arg if a normalizer is registered.