import inspect
import os
from importlib import import_module
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string, module_has_submodule
def _path_from_module(self, module):
    """Attempt to determine app's filesystem path from its module."""
    paths = list(getattr(module, '__path__', []))
    if len(paths) != 1:
        filename = getattr(module, '__file__', None)
        if filename is not None:
            paths = [os.path.dirname(filename)]
        else:
            paths = list(set(paths))
    if len(paths) > 1:
        raise ImproperlyConfigured("The app module %r has multiple filesystem locations (%r); you must configure this app with an AppConfig subclass with a 'path' class attribute." % (module, paths))
    elif not paths:
        raise ImproperlyConfigured("The app module %r has no filesystem location, you must configure this app with an AppConfig subclass with a 'path' class attribute." % module)
    return paths[0]