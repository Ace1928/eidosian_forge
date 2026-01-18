from functools import wraps
import pennylane as qml
from pennylane.math import conj, moveaxis, transpose
from pennylane.operation import Observable, Operation, Operator
from pennylane.queuing import QueuingManager
from pennylane.tape import make_qscript
from pennylane.compiler import compiler
from pennylane.compiler.compiler import CompileError
from .symbolicop import SymbolicOp
class AdjointOperation(Adjoint, Operation):
    """This mixin class is dynamically added to an ``Adjoint`` instance if the provided base class
    is an ``Operation``.

    .. warning::
        This mixin class should never be initialized independent of ``Adjoint``.

    Overriding the dunder method ``__new__`` in ``Adjoint`` allows us to customize the creation of
    an instance and dynamically add in parent classes.

    .. note:: Once the ``Operation`` class does not contain any unique logic any more, this mixin
    class can be removed.
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    @property
    def name(self):
        return self._name

    @property
    def basis(self):
        return self.base.basis

    @property
    def control_wires(self):
        return self.base.control_wires

    def single_qubit_rot_angles(self):
        omega, theta, phi = self.base.single_qubit_rot_angles()
        return [-phi, -theta, -omega]

    @property
    def grad_method(self):
        return self.base.grad_method

    @property
    def grad_recipe(self):
        return self.base.grad_recipe

    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies

    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        return -1.0 * self.base.generator()