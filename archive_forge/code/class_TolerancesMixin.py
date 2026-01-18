from abc import ABCMeta
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT
class TolerancesMixin(metaclass=TolerancesMeta):
    """Mixin Class for tolerances"""

    @property
    def atol(self):
        """Default absolute tolerance parameter for float comparisons."""
        return self.__class__.atol

    @property
    def rtol(self):
        """Default relative tolerance parameter for float comparisons."""
        return self.__class__.rtol