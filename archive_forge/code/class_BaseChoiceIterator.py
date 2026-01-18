from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import islice, tee, zip_longest
from django.utils.functional import Promise
class BaseChoiceIterator:
    """Base class for lazy iterators for choices."""

    def __eq__(self, other):
        if isinstance(other, Iterable):
            return all((a == b for a, b in zip_longest(self, other, fillvalue=object())))
        return super().__eq__(other)

    def __getitem__(self, index):
        if index < 0:
            return list(self)[index]
        try:
            return next(islice(self, index, index + 1))
        except StopIteration:
            raise IndexError('index out of range') from None

    def __iter__(self):
        raise NotImplementedError('BaseChoiceIterator subclasses must implement __iter__().')