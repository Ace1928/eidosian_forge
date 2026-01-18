import abc
import io
import os
from typing import Any, BinaryIO, Iterable, Iterator, NoReturn, Text, Optional
from typing import runtime_checkable, Protocol
from typing import Union
class ResourceReader(metaclass=abc.ABCMeta):
    """Abstract base class for loaders to provide resource reading support."""

    @abc.abstractmethod
    def open_resource(self, resource: Text) -> BinaryIO:
        """Return an opened, file-like object for binary reading.

        The 'resource' argument is expected to represent only a file name.
        If the resource cannot be found, FileNotFoundError is raised.
        """
        raise FileNotFoundError

    @abc.abstractmethod
    def resource_path(self, resource: Text) -> Text:
        """Return the file system path to the specified resource.

        The 'resource' argument is expected to represent only a file name.
        If the resource does not exist on the file system, raise
        FileNotFoundError.
        """
        raise FileNotFoundError

    @abc.abstractmethod
    def is_resource(self, path: Text) -> bool:
        """Return True if the named 'path' is a resource.

        Files are resources, directories are not.
        """
        raise FileNotFoundError

    @abc.abstractmethod
    def contents(self) -> Iterable[str]:
        """Return an iterable of entries in `package`."""
        raise FileNotFoundError