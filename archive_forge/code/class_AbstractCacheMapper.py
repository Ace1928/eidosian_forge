from __future__ import annotations
import abc
import hashlib
from fsspec.implementations.local import make_path_posix
class AbstractCacheMapper(abc.ABC):
    """Abstract super-class for mappers from remote URLs to local cached
    basenames.
    """

    @abc.abstractmethod
    def __call__(self, path: str) -> str:
        ...

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __hash__(self) -> int:
        return hash(type(self))