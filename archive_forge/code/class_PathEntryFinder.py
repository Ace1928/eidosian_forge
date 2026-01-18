from . import _bootstrap_external
from . import machinery
from ._abc import Loader
import abc
import warnings
from .resources.abc import ResourceReader, Traversable, TraversableResources
class PathEntryFinder(metaclass=abc.ABCMeta):
    """Abstract base class for path entry finders used by PathFinder."""

    def find_loader(self, fullname):
        """Return (loader, namespace portion) for the path entry.

        The fullname is a str.  The namespace portion is a sequence of
        path entries contributing to part of a namespace package. The
        sequence may be empty.  If loader is not None, the portion will
        be ignored.

        The portion will be discarded if another path entry finder
        locates the module as a normal module or package.

        This method is deprecated since Python 3.4 in favor of
        finder.find_spec(). If find_spec() is provided than backwards-compatible
        functionality is provided.
        """
        warnings.warn('PathEntryFinder.find_loader() is deprecated since Python 3.4 in favor of PathEntryFinder.find_spec() (available since 3.4)', DeprecationWarning, stacklevel=2)
        if not hasattr(self, 'find_spec'):
            return (None, [])
        found = self.find_spec(fullname)
        if found is not None:
            if not found.submodule_search_locations:
                portions = []
            else:
                portions = found.submodule_search_locations
            return (found.loader, portions)
        else:
            return (None, [])
    find_module = _bootstrap_external._find_module_shim

    def invalidate_caches(self):
        """An optional method for clearing the finder's cache, if any.
        This method is used by PathFinder.invalidate_caches().
        """