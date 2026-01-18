from __future__ import absolute_import
import types
from . import Errors
def Rep(re):
    """
    Rep(re) is an RE which matches zero or more repetitions of |re|.
    """
    result = Opt(Rep1(re))
    result.str = 'Rep(%s)' % re
    return result