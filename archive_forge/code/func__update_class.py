from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def _update_class(self, oldclass, newclass):
    """Update a class object."""
    olddict = oldclass.__dict__
    newdict = newclass.__dict__
    oldnames = set(olddict)
    newnames = set(newdict)
    for name in newnames - oldnames:
        setattr(oldclass, name, newdict[name])
        notify_info0('Added:', name, 'to', oldclass)
        self.found_change = True
    for name in (oldnames & newnames) - set(['__dict__', '__doc__']):
        self._update(oldclass, name, olddict[name], newdict[name], is_class_namespace=True)
    old_bases = getattr(oldclass, '__bases__', None)
    new_bases = getattr(newclass, '__bases__', None)
    if str(old_bases) != str(new_bases):
        notify_error('Changing the hierarchy of a class is not supported. %s may be inconsistent.' % (oldclass,))
    self._handle_namespace(oldclass, is_class_namespace=True)