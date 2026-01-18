from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def __test_ndim(self):
    self.parse_opts(u'int, 2')
    self.non_parse(ERR_BUF_NDIM, u"int, 'a'")
    self.non_parse(ERR_BUF_NDIM, u'int, -34')