from __future__ import (absolute_import, division, print_function)
def __load_module(self, fullname):
    """Load the requested module while avoiding infinite recursion."""
    self.loaded_modules.add(fullname)
    return import_module(fullname)