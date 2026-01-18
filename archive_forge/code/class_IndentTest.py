import textwrap
from oslotest import base
from oslo_policy import policy
from oslo_policy import sphinxext
class IndentTest(base.BaseTestCase):

    def test_indent(self):
        result = sphinxext._indent('foo\nbar')
        self.assertEqual('    foo\n    bar', result)
        result = sphinxext._indent('')
        self.assertEqual('', result)
        result = sphinxext._indent('\n')
        self.assertEqual('\n', result)
        result = sphinxext._indent('test\ntesting\n\nafter blank')
        self.assertEqual('    test\n    testing\n\n    after blank', result)
        result = sphinxext._indent('\tfoo\nbar')
        self.assertEqual('    \tfoo\n    bar', result)
        result = sphinxext._indent('    foo\nbar')
        self.assertEqual('        foo\n    bar', result)
        result = sphinxext._indent('foo\n    bar')
        self.assertEqual('    foo\n        bar', result)
        result = sphinxext._indent('foo\n\n    bar')
        self.assertEqual('    foo\n\n        bar', result)
        self.assertRaises(AttributeError, sphinxext._indent, None)