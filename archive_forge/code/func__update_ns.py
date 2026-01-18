from __future__ import unicode_literals
import sys
import logging
import re
import time
import xml.dom.minidom
from . import __author__, __copyright__, __license__, __version__
from .helpers import TYPE_MAP, TYPE_MARSHAL_FN, TYPE_UNMARSHAL_FN, \
def _update_ns(self, name):
    """Replace the defined namespace alias with tohse used by the client."""
    pref = self.__ns_rx.search(name)
    if pref:
        pref = pref.groups()[0]
        try:
            name = name.replace(pref, self.__namespaces_map[pref])
        except KeyError:
            log.warning('Unknown namespace alias %s' % name)
    return name