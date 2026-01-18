from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
class TestBufferParsing(CythonTest):

    def parse(self, s):
        return self.should_not_fail(lambda: self.fragment(s)).root

    def not_parseable(self, expected_error, s):
        e = self.should_fail(lambda: self.fragment(s), Errors.CompileError)
        self.assertEqual(expected_error, e.message_only)

    def test_basic(self):
        t = self.parse(u'cdef object[float, 4, ndim=2, foo=foo] x')
        bufnode = t.stats[0].base_type
        self.assertTrue(isinstance(bufnode, TemplatedTypeNode))
        self.assertEqual(2, len(bufnode.positional_args))

    def test_type_pos(self):
        self.parse(u'cdef object[short unsigned int, 3] x')

    def test_type_keyword(self):
        self.parse(u'cdef object[foo=foo, dtype=short unsigned int] x')

    def test_pos_after_key(self):
        self.not_parseable('Non-keyword arg following keyword arg', u'cdef object[foo=1, 2] x')