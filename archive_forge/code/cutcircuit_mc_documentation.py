import inspect
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.measurements import SampleMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from .cutstrategy import CutStrategy
from .kahypar import kahypar_cut
from .processing import qcut_processing_fn_mc, qcut_processing_fn_sample
from .tapes import _qcut_expand_fn, graph_to_tape, tape_to_graph
from .utils import (

    Expands fragment tapes into a sequence of random configurations of the contained pairs of
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    For each pair, a measurement is sampled from
    the Pauli basis and a state preparation is sampled from the corresponding pair of eigenstates.
    A settings array is also given which tracks the configuration pairs. Since each of the 4
    measurements has 2 possible eigenvectors, all configurations can be uniquely identified by
    8 values. The number of rows is determined by the number of cuts and the number of columns
    is determined by the number of shots.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`~.cut_circuit_mc` transform for more details.

    Args:
        tapes (Sequence[QuantumTape]): the fragment tapes containing :class:`MeasureNode` and
            :class:`PrepareNode` operations to be expanded
        communication_graph (nx.MultiDiGraph): the communication (quotient) graph of the fragmented
            full graph
        shots (int): number of shots

    Returns:
        Tuple[List[QuantumTape], np.ndarray]: the tapes corresponding to each configuration and the
        settings that track each configuration pair

    **Example**

    Consider the following circuit that contains a sample measurement:

    .. code-block:: python

        ops = [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.WireCut(wires=1),
            qml.CNOT(wires=[1, 2]),
        ]
        measurements = [qml.sample(wires=[0, 1, 2])]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can generate the fragment tapes using the following workflow:

    >>> g = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_nodes(g)
    >>> subgraphs, communication_graph = qml.qcut.fragment_graph(g)
    >>> tapes = [qml.qcut.graph_to_tape(sg) for sg in subgraphs]

    We can then expand over the measurement and preparation nodes to generate random
    configurations using:

    .. code-block:: python

        >>> configs, settings = qml.qcut.expand_fragment_tapes_mc(tapes, communication_graph, 3)
        >>> print(settings)
        [[1 6 2]]
        >>> for i, (c1, c2) in enumerate(zip(configs[0], configs[1])):
        ...     print(f"config {i}:")
        ...     print(c1.draw())
        ...     print("")
        ...     print(c2.draw())
        ...     print("")
        ...

        config 0:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Z]

        1: ──I─╭●─┤  Sample[|1⟩⟨1|]
        2: ────╰X─┤  Sample[|1⟩⟨1|]

        config 1:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Y]

        1: ──H──S─╭●─┤  Sample[|1⟩⟨1|]
        2: ───────╰X─┤  Sample[|1⟩⟨1|]

        config 2:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Y]

        1: ──X──H──S─╭●─┤  Sample[|1⟩⟨1|]
        2: ──────────╰X─┤  Sample[|1⟩⟨1|]

    