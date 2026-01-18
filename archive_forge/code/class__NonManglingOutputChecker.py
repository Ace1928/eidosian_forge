import doctest
import re
from ._impl import Mismatch
class _NonManglingOutputChecker(doctest.OutputChecker):
    """Doctest checker that works with unicode rather than mangling strings

    This is needed because current Python versions have tried to fix string
    encoding related problems, but regressed the default behaviour with
    unicode inputs in the process.

    In Python 2.6 and 2.7 ``OutputChecker.output_difference`` is was changed
    to return a bytestring encoded as per ``sys.stdout.encoding``, or utf-8 if
    that can't be determined. Worse, that encoding process happens in the
    innocent looking `_indent` global function. Because the
    `DocTestMismatch.describe` result may well not be destined for printing to
    stdout, this is no good for us. To get a unicode return as before, the
    method is monkey patched if ``doctest._encoding`` exists.

    Python 3 has a different problem. For some reason both inputs are encoded
    to ascii with 'backslashreplace', making an escaped string matches its
    unescaped form. Overriding the offending ``OutputChecker._toAscii`` method
    is sufficient to revert this.
    """

    def _toAscii(self, s):
        """Return ``s`` unchanged rather than mangling it to ascii"""
        return s
    if getattr(doctest, '_encoding', None) is not None:
        from types import FunctionType as __F
        __f = doctest.OutputChecker.output_difference.im_func
        __g = dict(__f.func_globals)

        def _indent(s, indent=4, _pattern=re.compile('^(?!$)', re.MULTILINE)):
            """Prepend non-empty lines in ``s`` with ``indent`` number of spaces"""
            return _pattern.sub(indent * ' ', s)
        __g['_indent'] = _indent
        output_difference = __F(__f.func_code, __g, 'output_difference')
        del __F, __f, __g, _indent