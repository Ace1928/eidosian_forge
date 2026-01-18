import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _getChild(self, elem_tag, elem, name, default=None):
    for prop in elem.findall(elem_tag):
        if prop.attrib['name'] == name:
            return self.convert(prop)
    else:
        return default