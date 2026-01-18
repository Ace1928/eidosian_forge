import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
class TestNonAsciiResults(TestCase):
    """Test all kinds of tracebacks are cleanly interpreted as unicode

    Currently only uses weak "contains" assertions, would be good to be much
    stricter about the expected output. This would add a few failures for the
    current release of IronPython for instance, which gets some traceback
    lines muddled.
    """
    _sample_texts = ('paɪθən', '南無', '§§§')
    _is_pypy = '__pypy__' in sys.builtin_module_names
    _error_on_character = os.name != 'java' and (not _is_pypy)

    def _run(self, stream, test):
        """Run the test, the same as in testtools.run but not to stdout"""
        result = TextTestResult(stream)
        result.startTestRun()
        try:
            return test.run(result)
        finally:
            result.stopTestRun()

    def _write_module(self, name, encoding, contents):
        """Create Python module on disk with contents in given encoding"""
        try:
            codecs.lookup(encoding)
        except LookupError:
            self.skipTest('Encoding unsupported by implementation: %r' % encoding)
        f = codecs.open(os.path.join(self.dir, name + '.py'), 'w', encoding)
        try:
            f.write(contents)
        finally:
            f.close()

    def _test_external_case(self, testline, coding='ascii', modulelevel='', suffix=''):
        """Create and run a test case in a separate module"""
        self._setup_external_case(testline, coding, modulelevel, suffix)
        return self._run_external_case()

    def _setup_external_case(self, testline, coding='ascii', modulelevel='', suffix=''):
        """Create a test case in a separate module"""
        _, prefix, self.modname = self.id().rsplit('.', 2)
        self.dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
        self.addCleanup(shutil.rmtree, self.dir)
        self._write_module(self.modname, coding, '# coding: %s\nimport testtools\n%s\nclass Test(testtools.TestCase):\n    def runTest(self):\n        %s\n' % (coding, modulelevel, testline))

    def _run_external_case(self):
        """Run the prepared test case in a separate module"""
        sys.path.insert(0, self.dir)
        self.addCleanup(sys.path.remove, self.dir)
        module = __import__(self.modname)
        self.addCleanup(sys.modules.pop, self.modname)
        stream = io.StringIO()
        self._run(stream, module.Test())
        return stream.getvalue()

    def _get_sample_text(self, encoding='unicode_internal'):
        if encoding is None:
            encoding = 'unicode_internal'
        for u in self._sample_texts:
            try:
                b = u.encode(encoding)
                if u == b.decode(encoding):
                    return (u, u)
            except (LookupError, UnicodeError):
                pass
        self.skipTest('Could not find a sample text for encoding: %r' % encoding)

    def _as_output(self, text):
        return text

    def test_non_ascii_failure_string(self):
        """Assertion contents can be non-ascii and should get decoded"""
        text, raw = self._get_sample_text(_get_exception_encoding())
        textoutput = self._test_external_case('self.fail(%s)' % ascii(raw))
        self.assertIn(self._as_output(text), textoutput)

    def test_non_ascii_failure_string_via_exec(self):
        """Assertion via exec can be non-ascii and still gets decoded"""
        text, raw = self._get_sample_text(_get_exception_encoding())
        textoutput = self._test_external_case(testline='exec ("self.fail(%s)")' % ascii(raw))
        self.assertIn(self._as_output(text), textoutput)

    def test_control_characters_in_failure_string(self):
        """Control characters in assertions should be escaped"""
        textoutput = self._test_external_case("self.fail('\\a\\a\\a')")
        self.expectFailure('Defense against the beeping horror unimplemented', self.assertNotIn, self._as_output('\x07\x07\x07'), textoutput)
        self.assertIn(self._as_output('���'), textoutput)

    def _local_os_error_matcher(self):
        return MatchesAny(Contains('FileExistsError: '), Contains('PermissionError: '))

    def test_os_error(self):
        """Locale error messages from the OS shouldn't break anything"""
        textoutput = self._test_external_case(modulelevel='import os', testline="os.mkdir('/')")
        self.assertThat(textoutput, self._local_os_error_matcher())

    def test_assertion_text_shift_jis(self):
        """A terminal raw backslash in an encoded string is weird but fine"""
        example_text = '十'
        textoutput = self._test_external_case(coding='shift_jis', testline="self.fail('%s')" % example_text)
        output_text = example_text
        self.assertIn(self._as_output('AssertionError: %s' % output_text), textoutput)

    def test_file_comment_iso2022_jp(self):
        """Control character escapes must be preserved if valid encoding"""
        example_text, _ = self._get_sample_text('iso2022_jp')
        textoutput = self._test_external_case(coding='iso2022_jp', testline="self.fail('Simple') # %s" % example_text)
        self.assertIn(self._as_output(example_text), textoutput)

    def test_unicode_exception(self):
        """Exceptions that can be formated losslessly as unicode should be"""
        example_text, _ = self._get_sample_text()
        exception_class = 'class FancyError(Exception):\n    def __unicode__(self):\n        return self.args[0]\n'
        textoutput = self._test_external_case(modulelevel=exception_class, testline='raise FancyError(%s)' % ascii(example_text))
        self.assertIn(self._as_output(example_text), textoutput)

    def test_unprintable_exception(self):
        """A totally useless exception instance still prints something"""
        exception_class = 'class UnprintableError(Exception):\n    def __str__(self):\n        raise RuntimeError\n    def __unicode__(self):\n        raise RuntimeError\n    def __repr__(self):\n        raise RuntimeError\n'
        if sys.version_info >= (3, 11):
            expected = 'UnprintableError: <exception str() failed>\n'
        else:
            expected = 'UnprintableError: <unprintable UnprintableError object>\n'
        textoutput = self._test_external_case(modulelevel=exception_class, testline='raise UnprintableError')
        self.assertIn(self._as_output(expected), textoutput)

    def test_non_ascii_dirname(self):
        """Script paths in the traceback can be non-ascii"""
        text, raw = self._get_sample_text(sys.getfilesystemencoding())
        textoutput = self._test_external_case(coding='utf-8', testline="self.fail('Simple')", suffix=raw)
        self.assertIn(self._as_output(text), textoutput)

    def test_syntax_error(self):
        """Syntax errors should still have fancy special-case formatting"""
        if platform.python_implementation() == 'PyPy':
            spaces = '         '
        elif sys.version_info >= (3, 10):
            spaces = '        '
        else:
            spaces = '          '
        marker = '^^^' if sys.version_info >= (3, 10) else '^'
        textoutput = self._test_external_case("exec ('f(a, b c)')")
        self.assertIn(self._as_output('  File "<string>", line 1\n    f(a, b c)\n' + ' ' * self._error_on_character + spaces + marker + '\nSyntaxError: '), textoutput)

    def test_syntax_error_malformed(self):
        """Syntax errors with bogus parameters should break anything"""
        textoutput = self._test_external_case('raise SyntaxError(3, 2, 1)')
        self.assertIn(self._as_output('\nSyntaxError: '), textoutput)

    def test_syntax_error_line_iso_8859_1(self):
        """Syntax error on a latin-1 line shows the line decoded"""
        text, raw = self._get_sample_text('iso-8859-1')
        textoutput = self._setup_external_case('import bad')
        self._write_module('bad', 'iso-8859-1', '# coding: iso-8859-1\n! = 0 # %s\n' % text)
        textoutput = self._run_external_case()
        self.assertIn(self._as_output('    ! = 0 # %s\n    ^\nSyntaxError: ' % (text,)), textoutput)

    def test_syntax_error_line_iso_8859_5(self):
        """Syntax error on a iso-8859-5 line shows the line decoded"""
        text, raw = self._get_sample_text('iso-8859-5')
        textoutput = self._setup_external_case('import bad')
        self._write_module('bad', 'iso-8859-5', '# coding: iso-8859-5\n%% = 0 # %s\n' % text)
        textoutput = self._run_external_case()
        self.assertThat(textoutput, MatchesRegex(self._as_output(('.*%% = 0 # %s\n' + ' ' * self._error_on_character + '\\s*\\^\nSyntaxError:.*') % (text,)), re.MULTILINE | re.DOTALL))

    def test_syntax_error_line_euc_jp(self):
        """Syntax error on a euc_jp line shows the line decoded"""
        text, raw = self._get_sample_text('euc_jp')
        textoutput = self._setup_external_case('import bad')
        self._write_module('bad', 'euc_jp', '# coding: euc_jp\n$ = 0 # %s\n' % text)
        textoutput = self._run_external_case()
        if self._is_pypy:
            self._error_on_character = True
        self.assertIn(self._as_output(('    $ = 0 # %s\n' + ' ' * self._error_on_character + '   ^\nSyntaxError: ') % (text,)), textoutput)

    def test_syntax_error_line_utf_8(self):
        """Syntax error on a utf-8 line shows the line decoded"""
        text, raw = self._get_sample_text('utf-8')
        textoutput = self._setup_external_case('import bad')
        self._write_module('bad', 'utf-8', '\ufeff^ = 0 # %s\n' % text)
        textoutput = self._run_external_case()
        if sys.version_info >= (3, 9):
            textoutput = textoutput.replace('\ufeff', '')
        self.assertThat(textoutput, MatchesRegex(self._as_output(('.*bad.py", line 1\n\\s*\\^ = 0 # %s\n' + ' ' * self._error_on_character + '\\s*\\^\nSyntaxError:.*') % text), re.M | re.S))