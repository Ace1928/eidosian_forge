from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def _check_variable_id(self, variable_id: int) -> None:
    if variable_id not in self.variables:
        raise model_storage.BadVariableIdError(variable_id)