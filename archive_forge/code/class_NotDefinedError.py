import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class NotDefinedError(IRError):
    """
    An undefined variable is encountered during interpretation of IR.
    """

    def __init__(self, name, loc=None):
        self.name = name
        msg = "The compiler failed to analyze the bytecode. Variable '%s' is not defined." % name
        super(NotDefinedError, self).__init__(msg, loc=loc)