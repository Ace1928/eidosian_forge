import heapq
from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane.transforms import transform
def _merge_no_duplicates(*iterables):
    """Merge K list without duplicate using python heapq ordered merging.

    Args:
        *iterables: A list of k sorted lists.

    Yields:
        Iterator: List from the merging of the k ones (without duplicates).
    """
    last = object()
    for val in heapq.merge(*iterables):
        if val != last:
            last = val
            yield val