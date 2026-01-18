import bisect
from collections import Counter, defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from nltk.internals import raise_unorderable_types, slice_bounds
class LazyMap(AbstractLazySequence):
    """
    A lazy sequence whose elements are formed by applying a given
    function to each element in one or more underlying lists.  The
    function is applied lazily -- i.e., when you read a value from the
    list, ``LazyMap`` will calculate that value by applying its
    function to the underlying lists' value(s).  ``LazyMap`` is
    essentially a lazy version of the Python primitive function
    ``map``.  In particular, the following two expressions are
    equivalent:

        >>> from nltk.collections import LazyMap
        >>> function = str
        >>> sequence = [1,2,3]
        >>> map(function, sequence) # doctest: +SKIP
        ['1', '2', '3']
        >>> list(LazyMap(function, sequence))
        ['1', '2', '3']

    Like the Python ``map`` primitive, if the source lists do not have
    equal size, then the value None will be supplied for the
    'missing' elements.

    Lazy maps can be useful for conserving memory, in cases where
    individual values take up a lot of space.  This is especially true
    if the underlying list's values are constructed lazily, as is the
    case with many corpus readers.

    A typical example of a use case for this class is performing
    feature detection on the tokens in a corpus.  Since featuresets
    are encoded as dictionaries, which can take up a lot of memory,
    using a ``LazyMap`` can significantly reduce memory usage when
    training and running classifiers.
    """

    def __init__(self, function, *lists, **config):
        """
        :param function: The function that should be applied to
            elements of ``lists``.  It should take as many arguments
            as there are ``lists``.
        :param lists: The underlying lists.
        :param cache_size: Determines the size of the cache used
            by this lazy map.  (default=5)
        """
        if not lists:
            raise TypeError('LazyMap requires at least two args')
        self._lists = lists
        self._func = function
        self._cache_size = config.get('cache_size', 5)
        self._cache = {} if self._cache_size > 0 else None
        self._all_lazy = sum((isinstance(lst, AbstractLazySequence) for lst in lists)) == len(lists)

    def iterate_from(self, index):
        if len(self._lists) == 1 and self._all_lazy:
            for value in self._lists[0].iterate_from(index):
                yield self._func(value)
            return
        elif len(self._lists) == 1:
            while True:
                try:
                    yield self._func(self._lists[0][index])
                except IndexError:
                    return
                index += 1
        elif self._all_lazy:
            iterators = [lst.iterate_from(index) for lst in self._lists]
            while True:
                elements = []
                for iterator in iterators:
                    try:
                        elements.append(next(iterator))
                    except:
                        elements.append(None)
                if elements == [None] * len(self._lists):
                    return
                yield self._func(*elements)
                index += 1
        else:
            while True:
                try:
                    elements = [lst[index] for lst in self._lists]
                except IndexError:
                    elements = [None] * len(self._lists)
                    for i, lst in enumerate(self._lists):
                        try:
                            elements[i] = lst[index]
                        except IndexError:
                            pass
                    if elements == [None] * len(self._lists):
                        return
                yield self._func(*elements)
                index += 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            sliced_lists = [lst[index] for lst in self._lists]
            return LazyMap(self._func, *sliced_lists)
        else:
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError('index out of range')
            if self._cache is not None and index in self._cache:
                return self._cache[index]
            try:
                val = next(self.iterate_from(index))
            except StopIteration as e:
                raise IndexError('index out of range') from e
            if self._cache is not None:
                if len(self._cache) > self._cache_size:
                    self._cache.popitem()
                self._cache[index] = val
            return val

    def __len__(self):
        return max((len(lst) for lst in self._lists))