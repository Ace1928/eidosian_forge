import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.block import IBlock, block, block_dict, block_list
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.piecewise_library.transforms import (
import pyomo.core.kernel.piecewise_library.transforms as transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import (
import pyomo.core.kernel.piecewise_library.transforms_nd as transforms_nd
import pyomo.core.kernel.piecewise_library.util as util
class Test_util(unittest.TestCase):

    def test_is_constant(self):
        self.assertEqual(util.is_constant([]), True)
        self.assertEqual(util.is_constant([1]), True)
        self.assertEqual(util.is_constant([1, 2]), False)
        self.assertEqual(util.is_constant([1, 1]), True)
        self.assertEqual(util.is_constant([1, 2, 3]), False)
        self.assertEqual(util.is_constant([2.1, 2.1, 2.1]), True)
        self.assertEqual(util.is_constant([1, 1, 3, 4]), False)
        self.assertEqual(util.is_constant([1, 1, 3, 3]), False)
        self.assertEqual(util.is_constant([1, 1, 1, 4]), False)
        self.assertEqual(util.is_constant([1, 1, 1, 1]), True)
        self.assertEqual(util.is_constant([-1, 1, 1, 1]), False)
        self.assertEqual(util.is_constant([1, -1, 1, 1]), False)
        self.assertEqual(util.is_constant([1, 1, -1, 1]), False)
        self.assertEqual(util.is_constant([1, 1, 1, -1]), False)

    def test_is_nondecreasing(self):
        self.assertEqual(util.is_nondecreasing([]), True)
        self.assertEqual(util.is_nondecreasing([1]), True)
        self.assertEqual(util.is_nondecreasing([1, 2]), True)
        self.assertEqual(util.is_nondecreasing([1, 2, 3]), True)
        self.assertEqual(util.is_nondecreasing([1, 1, 3, 4]), True)
        self.assertEqual(util.is_nondecreasing([1, 1, 3, 3]), True)
        self.assertEqual(util.is_nondecreasing([1, 1, 1, 4]), True)
        self.assertEqual(util.is_nondecreasing([1, 1, 1, 1]), True)
        self.assertEqual(util.is_nondecreasing([-1, 1, 1, 1]), True)
        self.assertEqual(util.is_nondecreasing([1, -1, 1, 1]), False)
        self.assertEqual(util.is_nondecreasing([1, 1, -1, 1]), False)
        self.assertEqual(util.is_nondecreasing([1, 1, 1, -1]), False)

    def test_is_nonincreasing(self):
        self.assertEqual(util.is_nonincreasing([]), True)
        self.assertEqual(util.is_nonincreasing([1]), True)
        self.assertEqual(util.is_nonincreasing([2, 1]), True)
        self.assertEqual(util.is_nonincreasing([3, 2, 1]), True)
        self.assertEqual(util.is_nonincreasing([4, 3, 2, 1]), True)
        self.assertEqual(util.is_nonincreasing([3, 3, 1, 1]), True)
        self.assertEqual(util.is_nonincreasing([4, 1, 1, 1]), True)
        self.assertEqual(util.is_nonincreasing([1, 1, 1, 1]), True)
        self.assertEqual(util.is_nonincreasing([-1, 1, 1, 1]), False)
        self.assertEqual(util.is_nonincreasing([1, -1, 1, 1]), False)
        self.assertEqual(util.is_nonincreasing([1, 1, -1, 1]), False)
        self.assertEqual(util.is_nonincreasing([1, 1, 1, -1]), True)

    def test_is_positive_power_of_two(self):
        self.assertEqual(util.is_positive_power_of_two(-8), False)
        self.assertEqual(util.is_positive_power_of_two(-4), False)
        self.assertEqual(util.is_positive_power_of_two(-3), False)
        self.assertEqual(util.is_positive_power_of_two(-2), False)
        self.assertEqual(util.is_positive_power_of_two(-1), False)
        self.assertEqual(util.is_positive_power_of_two(0), False)
        self.assertEqual(util.is_positive_power_of_two(1), True)
        self.assertEqual(util.is_positive_power_of_two(2), True)
        self.assertEqual(util.is_positive_power_of_two(3), False)
        self.assertEqual(util.is_positive_power_of_two(4), True)
        self.assertEqual(util.is_positive_power_of_two(5), False)
        self.assertEqual(util.is_positive_power_of_two(6), False)
        self.assertEqual(util.is_positive_power_of_two(7), False)
        self.assertEqual(util.is_positive_power_of_two(8), True)
        self.assertEqual(util.is_positive_power_of_two(15), False)
        self.assertEqual(util.is_positive_power_of_two(16), True)
        self.assertEqual(util.is_positive_power_of_two(31), False)
        self.assertEqual(util.is_positive_power_of_two(32), True)

    def test_log2floor(self):
        self.assertEqual(util.log2floor(1), 0)
        self.assertEqual(util.log2floor(2), 1)
        self.assertEqual(util.log2floor(3), 1)
        self.assertEqual(util.log2floor(4), 2)
        self.assertEqual(util.log2floor(5), 2)
        self.assertEqual(util.log2floor(6), 2)
        self.assertEqual(util.log2floor(7), 2)
        self.assertEqual(util.log2floor(8), 3)
        self.assertEqual(util.log2floor(9), 3)
        self.assertEqual(util.log2floor(2 ** 10), 10)
        self.assertEqual(util.log2floor(2 ** 10 + 1), 10)
        self.assertEqual(util.log2floor(2 ** 20), 20)
        self.assertEqual(util.log2floor(2 ** 20 + 1), 20)
        self.assertEqual(util.log2floor(2 ** 30), 30)
        self.assertEqual(util.log2floor(2 ** 30 + 1), 30)
        self.assertEqual(util.log2floor(2 ** 40), 40)
        self.assertEqual(util.log2floor(2 ** 40 + 1), 40)

    def test_generate_gray_code(self):
        self.assertEqual(util.generate_gray_code(0), [[]])
        self.assertEqual(util.generate_gray_code(1), [[0], [1]])
        self.assertEqual(util.generate_gray_code(2), [[0, 0], [0, 1], [1, 1], [1, 0]])
        self.assertEqual(util.generate_gray_code(3), [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]])
        self.assertEqual(util.generate_gray_code(4), [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]])

    def test_characterize_function(self):
        with self.assertRaises(ValueError):
            util.characterize_function([1, 2, -1], [1, 1, 1])
        fc, slopes = util.characterize_function([1, 2, 3], [1, 1, 1])
        self.assertEqual(fc, 1)
        self.assertEqual(slopes, [0, 0])
        fc, slopes = util.characterize_function([1, 2, 3], [1, 0, 1])
        self.assertEqual(fc, 2)
        self.assertEqual(slopes, [-1, 1])
        fc, slopes = util.characterize_function([1, 2, 3], [1, 2, 1])
        self.assertEqual(fc, 3)
        self.assertEqual(slopes, [1, -1])
        fc, slopes = util.characterize_function([1, 1, 2], [1, 2, 1])
        self.assertEqual(fc, 4)
        self.assertEqual(slopes, [None, -1])
        fc, slopes = util.characterize_function([1, 2, 3, 4], [1, 2, 1, 2])
        self.assertEqual(fc, 5)
        self.assertEqual(slopes, [1, -1, 1])

    @unittest.skipUnless(util.numpy_available and util.scipy_available, 'Numpy or Scipy is not available')
    def test_generate_delaunay(self):
        vlist = variable_list()
        vlist.append(variable(lb=0, ub=1))
        vlist.append(variable(lb=1, ub=2))
        vlist.append(variable(lb=2, ub=3))
        if not (util.numpy_available and util.scipy_available):
            with self.assertRaises(ImportError):
                util.generate_delaunay(vlist)
        else:
            tri = util.generate_delaunay(vlist, num=2)
            self.assertTrue(isinstance(tri, util.scipy.spatial.Delaunay))
            self.assertEqual(len(tri.simplices), 6)
            self.assertEqual(len(tri.points), 8)
            tri = util.generate_delaunay(vlist, num=3)
            self.assertTrue(isinstance(tri, util.scipy.spatial.Delaunay))
            self.assertEqual(len(tri.simplices), 62)
            self.assertEqual(len(tri.points), 27)
        vlist = variable_list()
        vlist.append(variable(lb=0))
        with self.assertRaises(ValueError):
            util.generate_delaunay(vlist)
        vlist = variable_list()
        vlist.append(variable(ub=0))
        with self.assertRaises(ValueError):
            util.generate_delaunay(vlist)