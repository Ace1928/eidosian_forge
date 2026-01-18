from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
def _set_sparse_double_vector(id_value_pairs: Iterable[Tuple[int, float]], proto: sparse_containers_pb2.SparseDoubleVectorProto) -> None:
    """id_value_pairs must be sorted, proto is filled."""
    if not id_value_pairs:
        return
    ids, values = zip(*id_value_pairs)
    proto.ids[:] = ids
    proto.values[:] = values