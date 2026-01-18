from contextlib import suppress
from io import TextIOWrapper
from . import abc
class ChildPath(abc.Traversable):
    """
        Path tied to a resource reader child.
        Can be read but doesn't expose any meaningful children.
        """

    def __init__(self, reader, name):
        self._reader = reader
        self._name = name

    def iterdir(self):
        return iter(())

    def is_file(self):
        return self._reader.is_resource(self.name)

    def is_dir(self):
        return not self.is_file()

    def joinpath(self, other):
        return CompatibilityFiles.OrphanPath(self.name, other)

    @property
    def name(self):
        return self._name

    def open(self, mode='r', *args, **kwargs):
        return _io_wrapper(self._reader.open_resource(self.name), mode, *args, **kwargs)