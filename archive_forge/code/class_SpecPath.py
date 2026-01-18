from contextlib import suppress
from io import TextIOWrapper
from . import abc
class SpecPath(abc.Traversable):
    """
        Path tied to a module spec.
        Can be read and exposes the resource reader children.
        """

    def __init__(self, spec, reader):
        self._spec = spec
        self._reader = reader

    def iterdir(self):
        if not self._reader:
            return iter(())
        return iter((CompatibilityFiles.ChildPath(self._reader, path) for path in self._reader.contents()))

    def is_file(self):
        return False
    is_dir = is_file

    def joinpath(self, other):
        if not self._reader:
            return CompatibilityFiles.OrphanPath(other)
        return CompatibilityFiles.ChildPath(self._reader, other)

    @property
    def name(self):
        return self._spec.name

    def open(self, mode='r', *args, **kwargs):
        return _io_wrapper(self._reader.open_resource(None), mode, *args, **kwargs)