from __future__ import annotations
import abc
from collections.abc import Callable, Hashable, Mapping, Sequence
from enum import Enum
from typing import (
class PostPersistCallable(Protocol[CollType_co]):
    """Protocol defining the signature of a ``__dask_postpersist__`` callable."""

    def __call__(self, dsk: Graph, *args: Any, rename: Mapping[str, str] | None=None) -> CollType_co:
        """Method called to rebuild a persisted collection.

        Parameters
        ----------
        dsk: Mapping
            A mapping which contains at least the output keys returned
            by __dask_keys__().
        *args : Any
            Additional optional arguments If no extra arguments are
            necessary, it must be an empty tuple.
        rename : Mapping[str, str], optional
            If defined, it indicates that output keys may be changing
            too; e.g. if the previous output of :meth:`__dask_keys__`
            was ``[("a", 0), ("a", 1)]``, after calling
            ``rebuild(dsk, *extra_args, rename={"a": "b"})``
            it must become ``[("b", 0), ("b", 1)]``.
            The ``rename`` mapping may not contain the collection
            name(s); in such case the associated keys do not change.
            It may contain replacements for unexpected names, which
            must be ignored.

        Returns
        -------
        Collection
            An equivalent Dask collection with the same keys as
            computed through a different graph.

        """
        raise NotImplementedError('Inheriting class must implement this method.')