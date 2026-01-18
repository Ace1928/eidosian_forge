from . import _bootstrap_external
from . import machinery
from ._abc import Loader
import abc
import warnings
from .resources.abc import ResourceReader, Traversable, TraversableResources
class MetaPathFinder(metaclass=abc.ABCMeta):
    """Abstract base class for import finders on sys.meta_path."""

    def find_module(self, fullname, path):
        """Return a loader for the module.

        If no module is found, return None.  The fullname is a str and
        the path is a list of strings or None.

        This method is deprecated since Python 3.4 in favor of
        finder.find_spec(). If find_spec() exists then backwards-compatible
        functionality is provided for this method.

        """
        warnings.warn('MetaPathFinder.find_module() is deprecated since Python 3.4 in favor of MetaPathFinder.find_spec() and is slated for removal in Python 3.12', DeprecationWarning, stacklevel=2)
        if not hasattr(self, 'find_spec'):
            return None
        found = self.find_spec(fullname, path)
        return found.loader if found is not None else None

    def invalidate_caches(self):
        """An optional method for clearing the finder's cache, if any.
        This method is used by importlib.invalidate_caches().
        """