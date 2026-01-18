from __future__ import absolute_import
import types
from . import Errors
def Opt(re):
    """
    Opt(re) is an RE which matches either |re| or the empty string.
    """
    result = Alt(re, Empty)
    result.str = 'Opt(%s)' % re
    return result