from __future__ import annotations
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import pathlib
import pickle
import types
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import partial
from numbers import Integral, Number
from operator import getitem
from typing import Any, Literal, TypeVar
import cloudpickle
from tlz import curry, groupby, identity, merge
from tlz.functoolz import Compose
from dask import config, local
from dask._compatibility import EMSCRIPTEN
from dask.core import flatten
from dask.core import get as simple_get
from dask.core import literal, quote
from dask.hashing import hash_buffer_hex
from dask.system import CPU_COUNT
from dask.typing import Key, SchedulerGetCallable
from dask.utils import (
class DaskMethodsMixin:
    """A mixin adding standard dask collection methods"""
    __slots__ = ()

    def visualize(self, filename='mydask', format=None, optimize_graph=False, **kwargs):
        """Render the computation of this object's task graph using graphviz.

        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str or None, optional
            The name of the file to write to disk. If the provided `filename`
            doesn't include an extension, '.png' will be used by default.
            If `filename` is None, no file will be written, and we communicate
            with dot using only pipes.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file.  Default is 'png'.
        optimize_graph : bool, optional
            If True, the graph is optimized before rendering.  Otherwise,
            the graph is displayed as is. Default is False.
        color: {None, 'order'}, optional
            Options to color nodes.  Provide ``cmap=`` keyword for additional
            colormap
        **kwargs
           Additional keyword arguments to forward to ``to_graphviz``.

        Examples
        --------
        >>> x.visualize(filename='dask.pdf')  # doctest: +SKIP
        >>> x.visualize(filename='dask.pdf', color='order')  # doctest: +SKIP

        Returns
        -------
        result : IPython.display.Image, IPython.display.SVG, or None
            See dask.dot.dot_graph for more information.

        See Also
        --------
        dask.visualize
        dask.dot.dot_graph

        Notes
        -----
        For more information on optimization see here:

        https://docs.dask.org/en/latest/optimize.html
        """
        return visualize(self, filename=filename, format=format, optimize_graph=optimize_graph, **kwargs)

    def persist(self, **kwargs):
        """Persist this dask collection into memory

        This turns a lazy Dask collection into a Dask collection with the same
        metadata, but now with the results fully computed or actively computing
        in the background.

        The action of function differs significantly depending on the active
        task scheduler.  If the task scheduler supports asynchronous computing,
        such as is the case of the dask.distributed scheduler, then persist
        will return *immediately* and the return value's task graph will
        contain Dask Future objects.  However if the task scheduler only
        supports blocking computation then the call to persist will *block*
        and the return value's task graph will contain concrete Python results.

        This function is particularly useful when using distributed systems,
        because the results will be kept in distributed memory, rather than
        returned to the local process as with compute.

        Parameters
        ----------
        scheduler : string, optional
            Which scheduler to use like "threads", "synchronous" or "processes".
            If not provided, the default is to check the global settings first,
            and then fall back to the collection defaults.
        optimize_graph : bool, optional
            If True [default], the graph is optimized before computation.
            Otherwise the graph is run as is. This can be useful for debugging.
        **kwargs
            Extra keywords to forward to the scheduler function.

        Returns
        -------
        New dask collections backed by in-memory data

        See Also
        --------
        dask.persist
        """
        result, = persist(self, traverse=False, **kwargs)
        return result

    def compute(self, **kwargs):
        """Compute this dask collection

        This turns a lazy Dask collection into its in-memory equivalent.
        For example a Dask array turns into a NumPy array and a Dask dataframe
        turns into a Pandas dataframe.  The entire dataset must fit into memory
        before calling this operation.

        Parameters
        ----------
        scheduler : string, optional
            Which scheduler to use like "threads", "synchronous" or "processes".
            If not provided, the default is to check the global settings first,
            and then fall back to the collection defaults.
        optimize_graph : bool, optional
            If True [default], the graph is optimized before computation.
            Otherwise the graph is run as is. This can be useful for debugging.
        kwargs
            Extra keywords to forward to the scheduler function.

        See Also
        --------
        dask.compute
        """
        result, = compute(self, traverse=False, **kwargs)
        return result

    def __await__(self):
        try:
            from distributed import futures_of, wait
        except ImportError as e:
            raise ImportError('Using async/await with dask requires the `distributed` package') from e
        from tornado import gen

        @gen.coroutine
        def f():
            if futures_of(self):
                yield wait(self)
            raise gen.Return(self)
        return f().__await__()