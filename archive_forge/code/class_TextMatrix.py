from typing import List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError
class TextMatrix:
    """Text representation of an array, with `__str__` method so it
    displays nicely in Jupyter notebooks"""

    def __init__(self, state, max_size=8, dims=None, prefix='', suffix=''):
        self.state = state
        self.max_size = max_size
        if dims is None:
            if isinstance(state, (Statevector, DensityMatrix)) and set(state.dims()) == {2} or (isinstance(state, Operator) and len(state.input_dims()) == len(state.output_dims()) and (set(state.input_dims()) == set(state.output_dims()) == {2})):
                dims = False
            else:
                dims = True
        self.dims = dims
        self.prefix = prefix
        self.suffix = suffix
        if isinstance(max_size, int):
            self.max_size = max_size
        elif isinstance(state, DensityMatrix):
            self.max_size = min(max_size) ** 2
        else:
            self.max_size = max_size[0]

    def __str__(self):
        threshold = self.max_size
        data = np.array2string(self.state._data, prefix=self.prefix, threshold=threshold, separator=',')
        dimstr = ''
        if self.dims:
            data += ',\n'
            dimstr += ' ' * len(self.prefix)
            if isinstance(self.state, (Statevector, DensityMatrix)):
                dimstr += f'dims={self.state._op_shape.dims_l()}'
            else:
                dimstr += f'input_dims={self.state.input_dims()}, '
                dimstr += f'output_dims={self.state.output_dims()}'
        return self.prefix + data + dimstr + self.suffix

    def __repr__(self):
        return self.__str__()