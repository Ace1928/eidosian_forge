from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol
class CacheStorage(Protocol):
    """Cache storage protocol, that should be implemented by the concrete cache storages.
    Used to store cached values for a single `@st.cache_data` decorated function
    serialized as bytes.

    CacheStorage instances should be created by `CacheStorageManager.create()` method.

    Notes
    -----
    Threading: The methods of this protocol could be called from multiple threads.
    This is a responsibility of the concrete implementation to ensure thread safety
    guarantees.
    """

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Returns the stored value for the key.

        Raises
        ------
        CacheStorageKeyNotFoundError
            Raised if the key is not in the storage.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        """Sets the value for a given key"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a given key"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all keys for the storage"""
        raise NotImplementedError

    def close(self) -> None:
        """Closes the cache storage, it is optional to implement, and should be used
        to close open resources, before we delete the storage instance.
        e.g. close the database connection etc.
        """
        pass