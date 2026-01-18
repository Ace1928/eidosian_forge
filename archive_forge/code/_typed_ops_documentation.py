from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Any, Callable, overload
from xarray.core import nputils, ops
from xarray.core.types import (
Mixin classes with arithmetic operators.