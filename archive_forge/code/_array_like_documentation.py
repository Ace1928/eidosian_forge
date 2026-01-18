from __future__ import annotations
import sys
from collections.abc import Collection, Callable, Sequence
from typing import Any, Protocol, Union, TypeVar, runtime_checkable
from numpy import (
from ._nested_sequence import _NestedSequence
A protocol class representing `~class.__array_function__`.