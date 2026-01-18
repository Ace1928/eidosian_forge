import warnings
from llvmlite import ir
from numba.cuda.cudadrv import nvvm, runtime
from numba.cuda.testing import unittest
from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
from numba.cuda.testing import skip_on_cudasim
@skip_on_cudasim('NVVM Driver unsupported in the simulator')
class TestArchOption(unittest.TestCase):

    def test_get_arch_option(self):
        self.assertEqual(nvvm.get_arch_option(5, 3), 'compute_53')
        self.assertEqual(nvvm.get_arch_option(7, 5), 'compute_75')
        self.assertEqual(nvvm.get_arch_option(7, 7), 'compute_75')
        supported_cc = nvvm.get_supported_ccs()
        for arch in supported_cc:
            self.assertEqual(nvvm.get_arch_option(*arch), 'compute_%d%d' % arch)
        self.assertEqual(nvvm.get_arch_option(1000, 0), 'compute_%d%d' % supported_cc[-1])