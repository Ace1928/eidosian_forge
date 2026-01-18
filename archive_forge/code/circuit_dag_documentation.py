from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
Finds all nodes before blocking ones.

        Args:
            is_blocker: The predicate that indicates whether or not an
            operation is blocking.
        