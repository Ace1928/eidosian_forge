from typing import Any, Iterable, Mapping
import networkx as nx
import cirq
from cirq_aqt import aqt_target_gateset
Return the maximum duration of the specified gate operation.

        Args:
            operation: The `cirq.Operation` for which to determine its duration.

        Raises:
            ValueError: if the operation has an unsupported gate type.
        