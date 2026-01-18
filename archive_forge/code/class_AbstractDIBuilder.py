import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
class AbstractDIBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def mark_variable(self, builder, allocavalue, name, lltype, size, line, datamodel=None, argidx=None):
        """Emit debug info for the variable.
        """
        pass

    @abc.abstractmethod
    def mark_location(self, builder, line):
        """Emit source location information to the given IRBuilder.
        """
        pass

    @abc.abstractmethod
    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        """Emit source location information for the given function.
        """
        pass

    @abc.abstractmethod
    def initialize(self):
        """Initialize the debug info. An opportunity for the debuginfo to
        prepare any necessary data structures.
        """

    @abc.abstractmethod
    def finalize(self):
        """Finalize the debuginfo by emitting all necessary metadata.
        """
        pass