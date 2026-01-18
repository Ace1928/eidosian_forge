import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
class DummyDIBuilder(AbstractDIBuilder):

    def __init__(self, module, filepath, cgctx, directives_only):
        pass

    def mark_variable(self, builder, allocavalue, name, lltype, size, line, datamodel=None, argidx=None):
        pass

    def mark_location(self, builder, line):
        pass

    def mark_subprogram(self, function, qualname, argnames, argtypes, line):
        pass

    def initialize(self):
        pass

    def finalize(self):
        pass