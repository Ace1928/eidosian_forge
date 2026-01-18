from abc import ABC, abstractmethod
from pennylane.operation import Operation
class AlgorithmicError(ABC):
    """Abstract base class representing an abstract type of error.
    This class can be used to create objects that track and propagate errors introduced by approximations and other algorithmic inaccuracies.

    Args:
        error (float): The numerical value of the error

    .. note::
        Child classes must implement the :func:`~.AlgorithmicError.combine` method which combines two
        instances of this error type (as if the associated gates were applied in series).
    """

    def __init__(self, error: float):
        self.error = error

    @abstractmethod
    def combine(self, other):
        """A method to combine two errors of the same type.
        (e.g., additive, square additive, multiplicative, etc.)

        Args:
            other (AlgorithmicError): The other instance of error being combined.

        Returns:
            AlgorithmicError: The total error after combination.
        """

    @staticmethod
    def get_error(approximate_op, exact_op, **kwargs):
        """A method to allow users to compute this type of error between two operators.

        Args:
            approximate_op (.Operator): The approximate operator.
            exact_op (.Operator): The exact operator.

        Returns:
            float: The error between the exact operator and its
            approximation.
        """
        raise NotImplementedError