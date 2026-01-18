from pygments.formatter import Formatter
from pygments.util import OptionError, get_choice_opt
from pygments.token import Token
from pygments.console import colorize
class TestcaseFormatter(Formatter):
    """
    Format tokens as appropriate for a new testcase.

    .. versionadded:: 2.0
    """
    name = 'Testcase'
    aliases = ['testcase']

    def __init__(self, **options):
        Formatter.__init__(self, **options)
        if self.encoding is not None and self.encoding != 'utf-8':
            raise ValueError('Only None and utf-8 are allowed encodings.')

    def format(self, tokensource, outfile):
        indentation = ' ' * 12
        rawbuf = []
        outbuf = []
        for ttype, value in tokensource:
            rawbuf.append(value)
            outbuf.append('%s(%s, %r),\n' % (indentation, ttype, value))
        before = TESTCASE_BEFORE % (u''.join(rawbuf),)
        during = u''.join(outbuf)
        after = TESTCASE_AFTER
        if self.encoding is None:
            outfile.write(before + during + after)
        else:
            outfile.write(before.encode('utf-8'))
            outfile.write(during.encode('utf-8'))
            outfile.write(after.encode('utf-8'))
        outfile.flush()