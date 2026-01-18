from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def __test_basic(self):
    buf = self.parse_opts(u'unsigned short int, 3')
    self.assertTrue(isinstance(buf.dtype_node, CSimpleBaseTypeNode))
    self.assertTrue(buf.dtype_node.signed == 0 and buf.dtype_node.longness == -1)
    self.assertEqual(3, buf.ndim)