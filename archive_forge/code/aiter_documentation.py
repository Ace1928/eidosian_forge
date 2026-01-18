from collections import deque
from typing import (

    Create ``n`` separate asynchronous iterators over ``iterable``

    This splits a single ``iterable`` into multiple iterators, each providing
    the same items in the same order.
    All child iterators may advance separately but share the same items
    from ``iterable`` -- when the most advanced iterator retrieves an item,
    it is buffered until the least advanced iterator has yielded it as well.
    A ``tee`` works lazily and can handle an infinite ``iterable``, provided
    that all iterators advance.

    .. code-block:: python3

        async def derivative(sensor_data):
            previous, current = a.tee(sensor_data, n=2)
            await a.anext(previous)  # advance one iterator
            return a.map(operator.sub, previous, current)

    Unlike :py:func:`itertools.tee`, :py:func:`~.tee` returns a custom type instead
    of a :py:class:`tuple`. Like a tuple, it can be indexed, iterated and unpacked
    to get the child iterators. In addition, its :py:meth:`~.tee.aclose` method
    immediately closes all children, and it can be used in an ``async with`` context
    for the same effect.

    If ``iterable`` is an iterator and read elsewhere, ``tee`` will *not*
    provide these items. Also, ``tee`` must internally buffer each item until the
    last iterator has yielded it; if the most and least advanced iterator differ
    by most data, using a :py:class:`list` is more efficient (but not lazy).

    If the underlying iterable is concurrency safe (``anext`` may be awaited
    concurrently) the resulting iterators are concurrency safe as well. Otherwise,
    the iterators are safe if there is only ever one single "most advanced" iterator.
    To enforce sequential use of ``anext``, provide a ``lock``
    - e.g. an :py:class:`asyncio.Lock` instance in an :py:mod:`asyncio` application -
    and access is automatically synchronised.
    