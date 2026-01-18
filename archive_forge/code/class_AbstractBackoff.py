import random
from abc import ABC, abstractmethod
class AbstractBackoff(ABC):
    """Backoff interface"""

    def reset(self):
        """
        Reset internal state before an operation.
        `reset` is called once at the beginning of
        every call to `Retry.call_with_retry`
        """
        pass

    @abstractmethod
    def compute(self, failures):
        """Compute backoff in seconds upon failure"""
        pass