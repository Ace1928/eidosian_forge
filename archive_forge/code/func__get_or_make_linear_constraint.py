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
def _get_or_make_linear_constraint(self, linear_constraint_id: int) -> LinearConstraint:
    result = self._linear_constraint_ids.get(linear_constraint_id)
    if result:
        return result
    result = LinearConstraint(self, linear_constraint_id)
    self._linear_constraint_ids[linear_constraint_id] = result
    return result