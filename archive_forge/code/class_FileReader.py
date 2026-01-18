import collections
import operator
import pathlib
import zipfile
from . import abc
from ._itertools import unique_everseen
class FileReader(abc.TraversableResources):

    def __init__(self, loader):
        self.path = pathlib.Path(loader.path).parent

    def resource_path(self, resource):
        """
        Return the file system path to prevent
        `resources.path()` from creating a temporary
        copy.
        """
        return str(self.path.joinpath(resource))

    def files(self):
        return self.path