from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
class TestDMMChaining(unittest.TestCase):

    def test_basic(self):
        dmm = DataModelManager()

        class int_handler(OpaqueModel):
            pass

        class float_handler(OpaqueModel):
            pass
        dmm.register(types.Integer, int_handler)
        dmm.register(types.Float, float_handler)
        inter_dmm = DataModelManager()

        class new_int_handler(OpaqueModel):
            pass
        inter_dmm.register(types.Integer, new_int_handler)
        chained_dmm = inter_dmm.chain(dmm)
        self.assertIsInstance(chained_dmm.lookup(types.intp), new_int_handler)
        self.assertNotIsInstance(chained_dmm.lookup(types.intp), int_handler)
        self.assertIsInstance(dmm.lookup(types.intp), int_handler)
        self.assertIsInstance(chained_dmm.lookup(types.float32), float_handler)
        self.assertIsInstance(dmm.lookup(types.float32), float_handler)
        self.assertIsInstance(inter_dmm.lookup(types.intp), new_int_handler)
        with self.assertRaises(KeyError):
            inter_dmm.lookup(types.float32)