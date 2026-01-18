import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
class PasteTestCase(TestCase):
    """Multiple tests for clipboard pasting"""

    def paste(self, txt, flags='-q'):
        """Paste input text, by default in quiet mode"""
        ip.hooks.clipboard_get = lambda: txt
        ip.run_line_magic('paste', flags)

    def setUp(self):
        self.original_clip = ip.hooks.clipboard_get

    def tearDown(self):
        ip.hooks.clipboard_get = self.original_clip

    def test_paste(self):
        ip.user_ns.pop('x', None)
        self.paste('x = 1')
        self.assertEqual(ip.user_ns['x'], 1)
        ip.user_ns.pop('x')

    def test_paste_pyprompt(self):
        ip.user_ns.pop('x', None)
        self.paste('>>> x=2')
        self.assertEqual(ip.user_ns['x'], 2)
        ip.user_ns.pop('x')

    def test_paste_py_multi(self):
        self.paste('\n        >>> x = [1,2,3]\n        >>> y = []\n        >>> for i in x:\n        ...     y.append(i**2)\n        ... \n        ')
        self.assertEqual(ip.user_ns['x'], [1, 2, 3])
        self.assertEqual(ip.user_ns['y'], [1, 4, 9])

    def test_paste_py_multi_r(self):
        """Now, test that self.paste -r works"""
        self.test_paste_py_multi()
        self.assertEqual(ip.user_ns.pop('x'), [1, 2, 3])
        self.assertEqual(ip.user_ns.pop('y'), [1, 4, 9])
        self.assertFalse('x' in ip.user_ns)
        ip.run_line_magic('paste', '-r')
        self.assertEqual(ip.user_ns['x'], [1, 2, 3])
        self.assertEqual(ip.user_ns['y'], [1, 4, 9])

    def test_paste_email(self):
        """Test pasting of email-quoted contents"""
        self.paste('        >> def foo(x):\n        >>     return x + 1\n        >> xx = foo(1.1)')
        self.assertEqual(ip.user_ns['xx'], 2.1)

    def test_paste_email2(self):
        """Email again; some programs add a space also at each quoting level"""
        self.paste('        > > def foo(x):\n        > >     return x + 1\n        > > yy = foo(2.1)     ')
        self.assertEqual(ip.user_ns['yy'], 3.1)

    def test_paste_email_py(self):
        """Email quoting of interactive input"""
        self.paste('        >> >>> def f(x):\n        >> ...   return x+1\n        >> ... \n        >> >>> zz = f(2.5)      ')
        self.assertEqual(ip.user_ns['zz'], 3.5)

    def test_paste_echo(self):
        """Also test self.paste echoing, by temporarily faking the writer"""
        w = StringIO()
        old_write = sys.stdout.write
        sys.stdout.write = w.write
        code = '\n        a = 100\n        b = 200'
        try:
            self.paste(code, '')
            out = w.getvalue()
        finally:
            sys.stdout.write = old_write
        self.assertEqual(ip.user_ns['a'], 100)
        self.assertEqual(ip.user_ns['b'], 200)
        assert out == code + '\n## -- End pasted text --\n'

    def test_paste_leading_commas(self):
        """Test multiline strings with leading commas"""
        tm = ip.magics_manager.registry['TerminalMagics']
        s = 'a = """\n,1,2,3\n"""'
        ip.user_ns.pop('foo', None)
        tm.store_or_execute(s, 'foo')
        self.assertIn('foo', ip.user_ns)

    def test_paste_trailing_question(self):
        """Test pasting sources with trailing question marks"""
        tm = ip.magics_manager.registry['TerminalMagics']
        s = "def funcfoo():\n   if True: #am i true?\n       return 'fooresult'\n"
        ip.user_ns.pop('funcfoo', None)
        self.paste(s)
        self.assertEqual(ip.user_ns['funcfoo'](), 'fooresult')