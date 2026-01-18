from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
class TestUInt64(test_factory()):
    fe_type = types.uint64