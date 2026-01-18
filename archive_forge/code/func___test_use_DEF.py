from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def __test_use_DEF(self):
    t = self.fragment(u'\n        DEF ndim = 3\n        def f():\n            cdef object[int, ndim] x\n            cdef object[ndim=ndim, dtype=int] y\n        ', pipeline=[NormalizeTree(self), PostParse(self)]).root
    stats = t.stats[0].body.stats
    self.assertTrue(stats[0].base_type.ndim == 3)
    self.assertTrue(stats[1].base_type.ndim == 3)