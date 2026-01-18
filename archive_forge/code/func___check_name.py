from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def __check_name(self, name):
    depr_map = {}
    depr_map[u'emph'] = u'em'
    if name in depr_map:
        msg = u"The tag '%s' is deprecated" % name
        msg += u", use '%s' instead." % depr_map[name]
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        return depr_map[name]
    return name