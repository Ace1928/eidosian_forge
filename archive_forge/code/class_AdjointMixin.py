import sys
from abc import ABC, abstractmethod
class AdjointMixin(ABC):
    """Abstract Mixin for operator adjoint and transpose operations.

    This class defines the following methods

        - :meth:`transpose`
        - :meth:`conjugate`
        - :meth:`adjoint`

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``conjugate(self)``
        - ``transpose(self)``
    """

    def adjoint(self) -> Self:
        """Return the adjoint of the CLASS."""
        return self.conjugate().transpose()

    @abstractmethod
    def conjugate(self) -> Self:
        """Return the conjugate of the CLASS."""

    @abstractmethod
    def transpose(self) -> Self:
        """Return the transpose of the CLASS."""