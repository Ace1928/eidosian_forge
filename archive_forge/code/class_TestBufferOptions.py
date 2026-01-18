from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
class TestBufferOptions(CythonTest):

    def nonfatal_error(self, error):
        self.error = error
        self.assertTrue(self.expect_error)

    def parse_opts(self, opts, expect_error=False):
        assert opts != ''
        s = u'def f():\n  cdef object[%s] x' % opts
        self.expect_error = expect_error
        root = self.fragment(s, pipeline=[NormalizeTree(self), PostParse(self)]).root
        if not expect_error:
            vardef = root.stats[0].body.stats[0]
            assert isinstance(vardef, CVarDefNode)
            buftype = vardef.base_type
            self.assertTrue(isinstance(buftype, TemplatedTypeNode))
            self.assertTrue(isinstance(buftype.base_type_node, CSimpleBaseTypeNode))
            self.assertEqual(u'object', buftype.base_type_node.name)
            return buftype
        else:
            self.assertTrue(len(root.stats[0].body.stats) == 0)

    def non_parse(self, expected_err, opts):
        self.parse_opts(opts, expect_error=True)
        self.assertEqual(expected_err, self.error.message_only)

    def __test_basic(self):
        buf = self.parse_opts(u'unsigned short int, 3')
        self.assertTrue(isinstance(buf.dtype_node, CSimpleBaseTypeNode))
        self.assertTrue(buf.dtype_node.signed == 0 and buf.dtype_node.longness == -1)
        self.assertEqual(3, buf.ndim)

    def __test_dict(self):
        buf = self.parse_opts(u'ndim=3, dtype=unsigned short int')
        self.assertTrue(isinstance(buf.dtype_node, CSimpleBaseTypeNode))
        self.assertTrue(buf.dtype_node.signed == 0 and buf.dtype_node.longness == -1)
        self.assertEqual(3, buf.ndim)

    def __test_ndim(self):
        self.parse_opts(u'int, 2')
        self.non_parse(ERR_BUF_NDIM, u"int, 'a'")
        self.non_parse(ERR_BUF_NDIM, u'int, -34')

    def __test_use_DEF(self):
        t = self.fragment(u'\n        DEF ndim = 3\n        def f():\n            cdef object[int, ndim] x\n            cdef object[ndim=ndim, dtype=int] y\n        ', pipeline=[NormalizeTree(self), PostParse(self)]).root
        stats = t.stats[0].body.stats
        self.assertTrue(stats[0].base_type.ndim == 3)
        self.assertTrue(stats[1].base_type.ndim == 3)