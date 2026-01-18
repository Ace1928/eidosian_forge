import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
class TestMagicRunSimple(tt.TempFileMixin):

    def test_simpledef(self):
        """Test that simple class definitions work."""
        src = 'class foo: pass\ndef f(): return foo()'
        self.mktmp(src)
        _ip.run_line_magic('run', str(self.fname))
        _ip.run_cell('t = isinstance(f(), foo)')
        assert _ip.user_ns['t'] is True

    @pytest.mark.xfail(platform.python_implementation() == 'PyPy', reason="expecting __del__ call on exit is unreliable and doesn't happen on PyPy")
    def test_obj_del(self):
        """Test that object's __del__ methods are called on exit."""
        src = "class A(object):\n    def __del__(self):\n        print('object A deleted')\na = A()\n"
        self.mktmp(src)
        err = None
        tt.ipexec_validate(self.fname, 'object A deleted', err)

    def test_aggressive_namespace_cleanup(self):
        """Test that namespace cleanup is not too aggressive GH-238

        Returning from another run magic deletes the namespace"""
        with tt.TempFileMixin() as empty:
            empty.mktmp('')
            src = 'ip = get_ipython()\nfor i in range(5):\n   try:\n       ip.magic(%r)\n   except NameError as e:\n       print(i)\n       break\n' % ('run ' + empty.fname)
            self.mktmp(src)
            _ip.run_line_magic('run', str(self.fname))
            _ip.run_cell('ip == get_ipython()')
            assert _ip.user_ns['i'] == 4

    def test_run_second(self):
        """Test that running a second file doesn't clobber the first, gh-3547"""
        self.mktmp('avar = 1\ndef afunc():\n  return avar\n')
        with tt.TempFileMixin() as empty:
            empty.mktmp('')
            _ip.run_line_magic('run', self.fname)
            _ip.run_line_magic('run', empty.fname)
            assert _ip.user_ns['afunc']() == 1

    def test_tclass(self):
        mydir = os.path.dirname(__file__)
        tc = os.path.join(mydir, 'tclass')
        src = f'import gc\n%run "{tc}" C-first\ngc.collect(0)\n%run "{tc}" C-second\ngc.collect(0)\n%run "{tc}" C-third\ngc.collect(0)\n%reset -f\n'
        self.mktmp(src, '.ipy')
        out = "ARGV 1-: ['C-first']\nARGV 1-: ['C-second']\ntclass.py: deleting object: C-first\nARGV 1-: ['C-third']\ntclass.py: deleting object: C-second\ntclass.py: deleting object: C-third\n"
        err = None
        tt.ipexec_validate(self.fname, out, err)

    def test_run_i_after_reset(self):
        """Check that %run -i still works after %reset (gh-693)"""
        src = 'yy = zz\n'
        self.mktmp(src)
        _ip.run_cell('zz = 23')
        try:
            _ip.run_line_magic('run', '-i %s' % self.fname)
            assert _ip.user_ns['yy'] == 23
        finally:
            _ip.run_line_magic('reset', '-f')
        _ip.run_cell('zz = 23')
        try:
            _ip.run_line_magic('run', '-i %s' % self.fname)
            assert _ip.user_ns['yy'] == 23
        finally:
            _ip.run_line_magic('reset', '-f')

    def test_unicode(self):
        """Check that files in odd encodings are accepted."""
        mydir = os.path.dirname(__file__)
        na = os.path.join(mydir, 'nonascii.py')
        _ip.magic('run "%s"' % na)
        assert _ip.user_ns['u'] == 'Ўт№Ф'

    def test_run_py_file_attribute(self):
        """Test handling of `__file__` attribute in `%run <file>.py`."""
        src = 't = __file__\n'
        self.mktmp(src)
        _missing = object()
        file1 = _ip.user_ns.get('__file__', _missing)
        _ip.run_line_magic('run', self.fname)
        file2 = _ip.user_ns.get('__file__', _missing)
        assert _ip.user_ns['t'] == self.fname
        assert file1 == file2

    def test_run_ipy_file_attribute(self):
        """Test handling of `__file__` attribute in `%run <file.ipy>`."""
        src = 't = __file__\n'
        self.mktmp(src, ext='.ipy')
        _missing = object()
        file1 = _ip.user_ns.get('__file__', _missing)
        _ip.run_line_magic('run', self.fname)
        file2 = _ip.user_ns.get('__file__', _missing)
        assert _ip.user_ns['t'] == self.fname
        assert file1 == file2

    def test_run_formatting(self):
        """ Test that %run -t -N<N> does not raise a TypeError for N > 1."""
        src = 'pass'
        self.mktmp(src)
        _ip.run_line_magic('run', '-t -N 1 %s' % self.fname)
        _ip.run_line_magic('run', '-t -N 10 %s' % self.fname)

    def test_ignore_sys_exit(self):
        """Test the -e option to ignore sys.exit()"""
        src = 'import sys; sys.exit(1)'
        self.mktmp(src)
        with tt.AssertPrints('SystemExit'):
            _ip.run_line_magic('run', self.fname)
        with tt.AssertNotPrints('SystemExit'):
            _ip.run_line_magic('run', '-e %s' % self.fname)

    def test_run_nb(self):
        """Test %run notebook.ipynb"""
        pytest.importorskip('nbformat')
        from nbformat import v4, writes
        nb = v4.new_notebook(cells=[v4.new_markdown_cell('The Ultimate Question of Everything'), v4.new_code_cell('answer=42')])
        src = writes(nb, version=4)
        self.mktmp(src, ext='.ipynb')
        _ip.run_line_magic('run', self.fname)
        assert _ip.user_ns['answer'] == 42

    def test_run_nb_error(self):
        """Test %run notebook.ipynb error"""
        pytest.importorskip('nbformat')
        from nbformat import v4, writes
        pytest.raises(Exception, _ip.magic, 'run')
        pytest.raises(Exception, _ip.magic, 'run foobar.ipynb')
        nb = v4.new_notebook(cells=[v4.new_code_cell('0/0')])
        src = writes(nb, version=4)
        self.mktmp(src, ext='.ipynb')
        pytest.raises(Exception, _ip.magic, 'run %s' % self.fname)

    def test_file_options(self):
        src = 'import sys\na = " ".join(sys.argv[1:])\n'
        self.mktmp(src)
        test_opts = '-x 3 --verbose'
        _ip.run_line_magic('run', '{0} {1}'.format(self.fname, test_opts))
        assert _ip.user_ns['a'] == test_opts