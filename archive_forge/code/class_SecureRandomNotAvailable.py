import os
import random
import warnings
class SecureRandomNotAvailable(RuntimeError):
    """
    Exception raised when no secure random algorithm is found.
    """