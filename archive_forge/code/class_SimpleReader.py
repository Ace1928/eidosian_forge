import abc
import io
import itertools
from typing import BinaryIO, List
from .abc import Traversable, TraversableResources
class SimpleReader(abc.ABC):
    """
    The minimum, low-level interface required from a resource
    provider.
    """

    @abc.abstractproperty
    def package(self):
        """
        The name of the package for which this reader loads resources.
        """

    @abc.abstractmethod
    def children(self):
        """
        Obtain an iterable of SimpleReader for available
        child containers (e.g. directories).
        """

    @abc.abstractmethod
    def resources(self):
        """
        Obtain available named resources for this virtual package.
        """

    @abc.abstractmethod
    def open_binary(self, resource):
        """
        Obtain a File-like for a named resource.
        """

    @property
    def name(self):
        return self.package.split('.')[-1]