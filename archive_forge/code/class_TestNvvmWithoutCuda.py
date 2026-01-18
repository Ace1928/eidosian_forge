from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import skip_on_cudasim
from numba.core import utils
from llvmlite import ir
from llvmlite import binding as llvm
import unittest
@skip_on_cudasim('libNVVM not supported in simulator')
@unittest.skipIf(utils.MACHINE_BITS == 32, 'CUDA not support for 32-bit')
@unittest.skipIf(not nvvm.is_available(), 'No libNVVM')
class TestNvvmWithoutCuda(unittest.TestCase):

    def test_nvvm_accepts_encoding(self):
        c = ir.Constant(ir.ArrayType(ir.IntType(8), 256), bytearray(range(256)))
        m = ir.Module()
        m.triple = 'nvptx64-nvidia-cuda'
        nvvm.add_ir_version(m)
        gv = ir.GlobalVariable(m, c.type, 'myconstant')
        gv.global_constant = True
        gv.initializer = c
        m.data_layout = nvvm.NVVM().data_layout
        parsed = llvm.parse_assembly(str(m))
        ptx = nvvm.llvm_to_ptx(str(parsed))
        elements = ', '.join([str(i) for i in range(256)])
        myconstant = f'myconstant[256] = {{{elements}}}'.encode('utf-8')
        self.assertIn(myconstant, ptx)