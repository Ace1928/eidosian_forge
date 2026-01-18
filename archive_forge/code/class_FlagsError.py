from abc import ABCMeta
from abc import abstractmethod
class FlagsError(ValueError):
    """Raised when a command line flag is not specified or is invalid."""
    pass