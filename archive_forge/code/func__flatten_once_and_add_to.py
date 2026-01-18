import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def _flatten_once_and_add_to(self, scale: float, processed_elements: _ProcessedElements, target_stack: _ToProcessElements) -> None:
    target_stack.append(self._linear, self._scalar * scale)