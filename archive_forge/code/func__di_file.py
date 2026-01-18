import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
def _di_file(self):
    return self.module.add_debug_info('DIFile', {'directory': os.path.dirname(self.filepath), 'filename': os.path.basename(self.filepath)})