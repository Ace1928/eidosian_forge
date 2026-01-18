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
class CustomQNode(qml.QNode):
    """
    A subclass with a custom __call__ method. The custom QNode transform returns an instance
    of this class.
    """

    def __call__(self, *args, **kwargs):
        shots = kwargs.pop('shots', False)
        shots = shots or self.device.shots
        if not shots:
            raise ValueError('A shots value must be provided in the device or when calling the QNode to be cut')
        if isinstance(shots, qml.measurements.Shots):
            shots = shots.total_shots
        qcut_tc = [tc for tc in self.transform_program if tc.transform.__name__ == 'cut_circuit_mc'][-1]
        qcut_tc._kwargs['shots'] = shots
        kwargs['shots'] = 1
        return super().__call__(*args, **kwargs)