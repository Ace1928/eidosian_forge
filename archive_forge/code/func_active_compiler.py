from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re
from semantic_version import Version
def active_compiler() -> Optional[str]:
    """Check which compiler is activated inside a :func:`~.qjit` evaluation context.

    This helper function may be used during implementation
    to allow differing logic for transformations or operations that are
    just-in-time compiled, versus those that are not.

    Returns:
        Optional[str]: Name of the active compiler inside a :func:`~.qjit` evaluation
        context. If there is no active compiler, ``None`` will be returned.

    **Example**

    This method can be used to execute logical
    branches that are conditioned on whether hybrid compilation with a specific
    compiler is occurring.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active_compiler() == "catalyst":
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.Z(0))

    >>> circuit(np.pi, np.pi / 2)
    1.0
    >>> qml.qjit(circuit)(np.pi, np.pi / 2)
    -1.0

    """
    for name, eps in AvailableCompilers.names_entrypoints.items():
        tracer_loader = eps['context'].load()
        if tracer_loader.is_tracing():
            return name
    return None