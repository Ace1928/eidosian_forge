import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _delayed_property(self, widget, prop):
    prop_value = self.convert(prop)
    if prop_value is not None:
        prop_name = prop.attrib['name']
        self.delayed_props.append((widget, False, 'set%s%s' % (ascii_upper(prop_name[0]), prop_name[1:]), prop_value))