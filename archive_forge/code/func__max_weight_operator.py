import functools
import itertools
from operator import mul
from typing import Dict, List, Iterable, Sequence, Set, Tuple, Union, cast
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil.experiment._main import Experiment
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, TensorProductState, _OneQState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.paulis import PauliTerm, sI
from pyquil.quil import Program
def _max_weight_operator(ops: Iterable[PauliTerm]) -> Union[None, PauliTerm]:
    """Construct a PauliTerm operator by taking the non-identity single-qubit operator at each
    qubit position.

    This function will return ``None`` if the input operators do not share a natural tensor
    product basis.

    For example, the max_weight_operator of ["XI", "IZ"] is "XZ". Asking for the max weight
    operator of something like ["XI", "ZI"] will return None.
    """
    mapping = dict()
    for op in ops:
        for idx, op_str in op:
            assert isinstance(idx, int)
            if idx in mapping:
                if mapping[idx] != op_str:
                    return None
            else:
                mapping[idx] = op_str
    op = functools.reduce(mul, (PauliTerm(op, q) for q, op in mapping.items()), sI())
    return op