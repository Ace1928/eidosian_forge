import contextlib
import copy
import operator
import pickle
import warnings
from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
class ScalarTest(parameterized.TestCase):

    @parameterized.product(scalar_type=INT4_TYPES)
    def testModuleName(self, scalar_type):
        self.assertEqual(scalar_type.__module__, 'ml_dtypes')

    @parameterized.product(scalar_type=INT4_TYPES)
    def testPickleable(self, scalar_type):
        x = np.arange(10, dtype=scalar_type)
        serialized = pickle.dumps(x)
        x_out = pickle.loads(serialized)
        self.assertEqual(x_out.dtype, x.dtype)
        np.testing.assert_array_equal(x_out.astype(int), x.astype(int))

    @parameterized.product(scalar_type=INT4_TYPES, python_scalar=[int, float])
    def testRoundTripToPythonScalar(self, scalar_type, python_scalar):
        for v in VALUES[scalar_type]:
            self.assertEqual(v, scalar_type(v))
            self.assertEqual(python_scalar(v), python_scalar(scalar_type(v)))
            self.assertEqual(scalar_type(v), scalar_type(python_scalar(scalar_type(v))))

    @parameterized.product(scalar_type=INT4_TYPES)
    def testRoundTripNumpyTypes(self, scalar_type):
        for dtype in [np.int8, np.int32]:
            for f in VALUES[scalar_type]:
                self.assertEqual(dtype(f), dtype(scalar_type(dtype(f))))
                self.assertEqual(int(dtype(f)), int(scalar_type(dtype(f))))
                self.assertEqual(dtype(f), dtype(scalar_type(np.array(f, dtype))))
            np.testing.assert_equal(dtype(np.array(VALUES[scalar_type], scalar_type)), np.array(VALUES[scalar_type], dtype))

    @parameterized.product(scalar_type=INT4_TYPES)
    def testStr(self, scalar_type):
        for value in VALUES[scalar_type]:
            self.assertEqual(str(value), str(scalar_type(value)))

    @parameterized.product(scalar_type=INT4_TYPES)
    def testRepr(self, scalar_type):
        for value in VALUES[scalar_type]:
            self.assertEqual(str(value), str(scalar_type(value)))

    @parameterized.product(scalar_type=INT4_TYPES)
    def testItem(self, scalar_type):
        self.assertIsInstance(scalar_type(3).item(), int)
        self.assertEqual(scalar_type(3).item(), 3)

    @parameterized.product(scalar_type=INT4_TYPES)
    def testHash(self, scalar_type):
        for v in VALUES[scalar_type]:
            self.assertEqual(hash(v), hash(scalar_type(v)), msg=v)

    @parameterized.product(scalar_type=INT4_TYPES, op=[operator.le, operator.lt, operator.eq, operator.ne, operator.ge, operator.gt])
    def testComparison(self, scalar_type, op):
        for v in VALUES[scalar_type]:
            for w in VALUES[scalar_type]:
                self.assertEqual(op(v, w), op(scalar_type(v), scalar_type(w)))

    @parameterized.product(scalar_type=INT4_TYPES, op=[operator.neg, operator.pos])
    def testUnop(self, scalar_type, op):
        for v in VALUES[scalar_type]:
            out = op(scalar_type(v))
            self.assertIsInstance(out, scalar_type)
            self.assertEqual(scalar_type(op(v)), out, msg=v)

    @parameterized.product(scalar_type=INT4_TYPES, op=[operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod])
    def testBinop(self, scalar_type, op):
        for v in VALUES[scalar_type]:
            for w in VALUES[scalar_type]:
                if w == 0 and op in [operator.floordiv, operator.mod]:
                    with self.assertRaises(ZeroDivisionError):
                        op(scalar_type(v), scalar_type(w))
                else:
                    out = op(scalar_type(v), scalar_type(w))
                    self.assertIsInstance(out, scalar_type)
                    self.assertEqual(scalar_type(op(v, w)), out, msg=(v, w))