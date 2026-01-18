import re
import six
from genshi.core import TEXT
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.directives import *
from genshi.template.interpolation import interpolate
def _get_delims(self):
    return self._delims