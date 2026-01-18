from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def _variables_to_proto(variables: Iterable[Tuple[int, _VariableStorage]], proto: model_pb2.VariablesProto) -> None:
    """Exports variables to proto."""
    has_named_var = False
    for _, var_storage in variables:
        if var_storage.name:
            has_named_var = True
            break
    for var_id, var_storage in variables:
        proto.ids.append(var_id)
        proto.lower_bounds.append(var_storage.lower_bound)
        proto.upper_bounds.append(var_storage.upper_bound)
        proto.integers.append(var_storage.is_integer)
        if has_named_var:
            proto.names.append(var_storage.name)