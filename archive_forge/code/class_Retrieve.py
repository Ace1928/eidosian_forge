from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, TypeVar
class Retrieve(Protocol[D]):
    """
    A retrieval callable, usable within a `Registry` for resource retrieval.

    Does not make assumptions about where the resource might be coming from.
    """

    def __call__(self, uri: URI) -> Resource[D]:
        """
        Retrieve the resource with the given URI.

        Raise `referencing.exceptions.NoSuchResource` if you wish to indicate
        the retriever cannot lookup the given URI.
        """
        ...