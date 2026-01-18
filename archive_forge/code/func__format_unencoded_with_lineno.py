import sys
from pygments.formatter import Formatter
from pygments.token import Keyword, Name, Comment, String, Error, \
from pygments.util import get_choice_opt
def _format_unencoded_with_lineno(self, tokensource, outfile):
    self._write_lineno(outfile)
    for ttype, value in tokensource:
        if value.endswith('\n'):
            self._write_lineno(outfile)
            value = value[:-1]
        color = self.colorscheme.get(ttype)
        while color is None:
            ttype = ttype[:-1]
            color = self.colorscheme.get(ttype)
        if color:
            color = color[self.darkbg]
            spl = value.split('\n')
            for line in spl[:-1]:
                self._write_lineno(outfile)
                if line:
                    outfile.write(ircformat(color, line[:-1]))
            if spl[-1]:
                outfile.write(ircformat(color, spl[-1]))
        else:
            outfile.write(value)
    outfile.write('\n')