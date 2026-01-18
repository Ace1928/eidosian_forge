import bisect
from collections import Counter, defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from nltk.internals import raise_unorderable_types, slice_bounds
class LazyConcatenation(AbstractLazySequence):
    """
    A lazy sequence formed by concatenating a list of lists.  This
    underlying list of lists may itself be lazy.  ``LazyConcatenation``
    maintains an index that it uses to keep track of the relationship
    between offsets in the concatenated lists and offsets in the
    sublists.
    """

    def __init__(self, list_of_lists):
        self._list = list_of_lists
        self._offsets = [0]

    def __len__(self):
        if len(self._offsets) <= len(self._list):
            for _ in self.iterate_from(self._offsets[-1]):
                pass
        return self._offsets[-1]

    def iterate_from(self, start_index):
        if start_index < self._offsets[-1]:
            sublist_index = bisect.bisect_right(self._offsets, start_index) - 1
        else:
            sublist_index = len(self._offsets) - 1
        index = self._offsets[sublist_index]
        if isinstance(self._list, AbstractLazySequence):
            sublist_iter = self._list.iterate_from(sublist_index)
        else:
            sublist_iter = islice(self._list, sublist_index, None)
        for sublist in sublist_iter:
            if sublist_index == len(self._offsets) - 1:
                assert index + len(sublist) >= self._offsets[-1], 'offsets not monotonic increasing!'
                self._offsets.append(index + len(sublist))
            else:
                assert self._offsets[sublist_index + 1] == index + len(sublist), 'inconsistent list value (num elts)'
            yield from sublist[max(0, start_index - index):]
            index += len(sublist)
            sublist_index += 1