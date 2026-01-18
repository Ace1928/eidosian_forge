from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_full(self, inner, outfile):
    if self.cssfile:
        if os.path.isabs(self.cssfile):
            cssfilename = self.cssfile
        else:
            try:
                filename = outfile.name
                if not filename or filename[0] == '<':
                    raise AttributeError
                cssfilename = os.path.join(os.path.dirname(filename), self.cssfile)
            except AttributeError:
                print('Note: Cannot determine output file name, using current directory as base for the CSS file name', file=sys.stderr)
                cssfilename = self.cssfile
        try:
            if not os.path.exists(cssfilename) or not self.noclobber_cssfile:
                cf = open(cssfilename, 'w')
                cf.write(CSSFILE_TEMPLATE % {'styledefs': self.get_style_defs('body')})
                cf.close()
        except IOError as err:
            err.strerror = 'Error writing CSS file: ' + err.strerror
            raise
        yield (0, DOC_HEADER_EXTERNALCSS % dict(title=self.title, cssfile=self.cssfile, encoding=self.encoding))
    else:
        yield (0, DOC_HEADER % dict(title=self.title, styledefs=self.get_style_defs('body'), encoding=self.encoding))
    for t, line in inner:
        yield (t, line)
    yield (0, DOC_FOOTER)