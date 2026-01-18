import re
import six
from genshi.core import TEXT
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.directives import *
from genshi.template.interpolation import interpolate
def _set_delims(self, delims):
    if len(delims) != 4:
        raise ValueError('delimiers tuple must have exactly four elements')
    self._delims = delims
    self._directive_re = re.compile(self._DIRECTIVE_RE % tuple([re.escape(d) for d in delims]), re.DOTALL)
    self._escape_re = re.compile(self._ESCAPE_RE % tuple([re.escape(d) for d in delims[::2]]))