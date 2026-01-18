import sys
from pygments.formatter import Formatter
from pygments.token import Keyword, Name, Comment, String, Error, \
from pygments.util import get_choice_opt
def _write_lineno(self, outfile):
    self._lineno += 1
    outfile.write('\n%04d: ' % self._lineno)