import pkgutil
import sys
from _pydev_bundle import pydev_log
def _iter_attr(self):
    for extension in self.loaded_extensions:
        dunder_all = getattr(extension, '__all__', None)
        for attr_name in dir(extension):
            if not attr_name.startswith('_'):
                if dunder_all is None or attr_name in dunder_all:
                    yield (attr_name, getattr(extension, attr_name))