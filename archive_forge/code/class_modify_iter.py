import collections
import warnings
from typing import Any, Iterable, Optional
from sphinx.deprecation import RemovedInSphinx70Warning
class modify_iter(peek_iter):
    """An iterator object that supports modifying items as they are returned.

    Parameters
    ----------
    o : iterable or callable
        `o` is interpreted very differently depending on the presence of
        `sentinel`.

        If `sentinel` is not given, then `o` must be a collection object
        which supports either the iteration protocol or the sequence protocol.

        If `sentinel` is given, then `o` must be a callable object.

    sentinel : any value, optional
        If given, the iterator will call `o` with no arguments for each
        call to its `next` method; if the value returned is equal to
        `sentinel`, :exc:`StopIteration` will be raised, otherwise the
        value will be returned.

    modifier : callable, optional
        The function that will be used to modify each item returned by the
        iterator. `modifier` should take a single argument and return a
        single value. Defaults to ``lambda x: x``.

        If `sentinel` is not given, `modifier` must be passed as a keyword
        argument.

    Attributes
    ----------
    modifier : callable
        `modifier` is called with each item in `o` as it is iterated. The
        return value of `modifier` is returned in lieu of the item.

        Values returned by `peek` as well as `next` are affected by
        `modifier`. However, `modify_iter.sentinel` is never passed through
        `modifier`; it will always be returned from `peek` unmodified.

    Example
    -------
    >>> a = ["     A list    ",
    ...      "   of strings  ",
    ...      "      with     ",
    ...      "      extra    ",
    ...      "   whitespace. "]
    >>> modifier = lambda s: s.strip().replace('with', 'without')
    >>> for s in modify_iter(a, modifier=modifier):
    ...   print('"%s"' % s)
    "A list"
    "of strings"
    "without"
    "extra"
    "whitespace."

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """__init__(o, sentinel=None, modifier=lambda x: x)"""
        if 'modifier' in kwargs:
            self.modifier = kwargs['modifier']
        elif len(args) > 2:
            self.modifier = args[2]
            args = args[:2]
        else:
            self.modifier = lambda x: x
        if not callable(self.modifier):
            raise TypeError('modify_iter(o, modifier): modifier must be callable')
        super().__init__(*args)

    def _fillcache(self, n: Optional[int]) -> None:
        """Cache `n` modified items. If `n` is 0 or None, 1 item is cached.

        Each item returned by the iterator is passed through the
        `modify_iter.modified` function before being cached.

        """
        if not n:
            n = 1
        try:
            while len(self._cache) < n:
                self._cache.append(self.modifier(next(self._iterable)))
        except StopIteration:
            while len(self._cache) < n:
                self._cache.append(self.sentinel)