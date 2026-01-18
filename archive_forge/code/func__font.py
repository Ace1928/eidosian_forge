import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _font(self, prop):
    newfont = self.factory.createQObject('QFont', 'font', (), is_attribute=False)
    for attr, converter in self._font_attributes:
        v = prop.findtext('./%s' % (attr.lower(),))
        if v is None:
            continue
        getattr(newfont, 'set%s' % (attr,))(converter(v))
    return newfont