import abc
import os.path
from contextlib import contextmanager
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.datamodel.models import ComplexModel, UniTupleModel
from numba.core import config
Util to create constant boolean in metadata
        