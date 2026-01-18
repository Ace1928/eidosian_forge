from . import _bootstrap_external
from . import machinery
from ._abc import Loader
import abc
import warnings
from .resources.abc import ResourceReader, Traversable, TraversableResources
class ResourceLoader(Loader):
    """Abstract base class for loaders which can return data from their
    back-end storage.

    This ABC represents one of the optional protocols specified by PEP 302.

    """

    @abc.abstractmethod
    def get_data(self, path):
        """Abstract method which when implemented should return the bytes for
        the specified path.  The path must be a str."""
        raise OSError