import warnings
import functools
from copy import copy
from functools import wraps
from inspect import signature
from typing import List
import numpy as np
from scipy import sparse
import pennylane as qml
from pennylane import operation
from pennylane import math as qmlmath
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane.compiler import compiler
from .symbolicop import SymbolicOp
from .controlled_decompositions import ctrl_decomp_bisect, ctrl_decomp_zyz
class ControlledOp(Controlled, operation.Operation):
    """Operation-specific methods and properties for the :class:`~.ops.op_math.Controlled` class.

    When an :class:`~.operation.Operation` is provided to the :class:`~.ops.op_math.Controlled`
    class, this type is constructed instead. It adds some additional :class:`~.operation.Operation`
    specific methods and properties.

    When we no longer rely on certain functionality through ``Operation``, we can get rid of this
    class.

    .. seealso:: :class:`~.Controlled`
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    def __init__(self, base, control_wires, control_values=None, work_wires=None, id=None):
        super().__init__(base, control_wires, control_values, work_wires, id)
        if self.grad_recipe is None:
            self.grad_recipe = [None] * self.num_params

    @property
    def name(self):
        return self._name

    @property
    def grad_method(self):
        return self.base.grad_method

    @property
    def parameter_frequencies(self):
        if self.base.num_params == 1:
            try:
                base_gen = qml.generator(self.base, format='observable')
            except operation.GeneratorUndefinedError as e:
                raise operation.ParameterFrequenciesUndefinedError(f'Operation {self.base.name} does not have parameter frequencies defined.') from e
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='.+ eigenvalues will be computed numerically\\.')
                base_gen_eigvals = qml.eigvals(base_gen, k=2 ** self.base.num_wires)
            gen_eigvals = np.append(base_gen_eigvals, 0)
            processed_gen_eigvals = tuple(np.round(gen_eigvals, 8))
            return [qml.gradients.eigvals_to_frequencies(processed_gen_eigvals)]
        raise operation.ParameterFrequenciesUndefinedError(f'Operation {self.name} does not have parameter frequencies defined, and parameter frequencies can not be computed via generator for more than one parameter.')