import sys
from abc import ABC, abstractmethod
class MultiplyMixin(ABC):
    """Abstract Mixin for scalar multiplication.

    This class defines the following operator overloads:

        - ``*`` / ``__rmul__`
        - ``/`` / ``__truediv__``
        - ``__neg__``

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``_multiply(self, other)``
    """

    def __rmul__(self, other) -> Self:
        return self._multiply(other)

    def __mul__(self, other) -> Self:
        return self._multiply(other)

    def __truediv__(self, other) -> Self:
        return self._multiply(1 / other)

    def __neg__(self) -> Self:
        return self._multiply(-1)

    @abstractmethod
    def _multiply(self, other):
        """Return the CLASS other * self.

        Args:
            other (complex): a complex number.

        Returns:
            CLASS: the CLASS other * self.
        """