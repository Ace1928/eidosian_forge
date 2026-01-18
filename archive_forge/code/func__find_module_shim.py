import _imp
import _io
import sys
import _warnings
import marshal
def _find_module_shim(self, fullname):
    """Try to find a loader for the specified module by delegating to
    self.find_loader().

    This method is deprecated in favor of finder.find_spec().

    """
    _warnings.warn('find_module() is deprecated and slated for removal in Python 3.12; use find_spec() instead', DeprecationWarning)
    loader, portions = self.find_loader(fullname)
    if loader is None and len(portions):
        msg = 'Not importing directory {}: missing __init__'
        _warnings.warn(msg.format(portions[0]), ImportWarning)
    return loader