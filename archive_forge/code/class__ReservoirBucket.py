import collections
import random
import threading
class _ReservoirBucket:
    """A container for items from a stream, that implements reservoir sampling.

    It always stores the most recent item as its final item.
    """

    def __init__(self, _max_size, _random=None, always_keep_last=True):
        """Create the _ReservoirBucket.

        Args:
          _max_size: The maximum size the reservoir bucket may grow to. If size is
            zero, the bucket has unbounded size.
          _random: The random number generator to use. If not specified, defaults to
            random.Random(0).
          always_keep_last: Whether the latest seen item should always be included
            in the end of the bucket.

        Raises:
          ValueError: if the size is not a nonnegative integer.
        """
        if _max_size < 0 or _max_size != round(_max_size):
            raise ValueError('_max_size must be nonnegative int, was %s' % _max_size)
        self.items = []
        self._mutex = threading.Lock()
        self._max_size = _max_size
        self._num_items_seen = 0
        if _random is not None:
            self._random = _random
        else:
            self._random = random.Random(0)
        self.always_keep_last = always_keep_last

    def AddItem(self, item, f=lambda x: x):
        """Add an item to the ReservoirBucket, replacing an old item if
        necessary.

        The new item is guaranteed to be added to the bucket, and to be the last
        element in the bucket. If the bucket has reached capacity, then an old item
        will be replaced. With probability (_max_size/_num_items_seen) a random item
        in the bucket will be popped out and the new item will be appended
        to the end. With probability (1 - _max_size/_num_items_seen)
        the last item in the bucket will be replaced.

        Since the O(n) replacements occur with O(1/_num_items_seen) likelihood,
        the amortized runtime is O(1).

        Args:
          item: The item to add to the bucket.
          f: A function to transform item before addition, if it will be kept in
            the reservoir.
        """
        with self._mutex:
            if len(self.items) < self._max_size or self._max_size == 0:
                self.items.append(f(item))
            else:
                r = self._random.randint(0, self._num_items_seen)
                if r < self._max_size:
                    self.items.pop(r)
                    self.items.append(f(item))
                elif self.always_keep_last:
                    self.items[-1] = f(item)
            self._num_items_seen += 1

    def FilterItems(self, filterFn):
        """Filter items in a ReservoirBucket, using a filtering function.

        Filtering items from the reservoir bucket must update the
        internal state variable self._num_items_seen, which is used for determining
        the rate of replacement in reservoir sampling. Ideally, self._num_items_seen
        would contain the exact number of items that have ever seen by the
        ReservoirBucket and satisfy filterFn. However, the ReservoirBucket does not
        have access to all items seen -- it only has access to the subset of items
        that have survived sampling (self.items). Therefore, we estimate
        self._num_items_seen by scaling it by the same ratio as the ratio of items
        not removed from self.items.

        Args:
          filterFn: A function that returns True for items to be kept.

        Returns:
          The number of items removed from the bucket.
        """
        with self._mutex:
            size_before = len(self.items)
            self.items = list(filter(filterFn, self.items))
            size_diff = size_before - len(self.items)
            prop_remaining = len(self.items) / float(size_before) if size_before > 0 else 0
            self._num_items_seen = int(round(self._num_items_seen * prop_remaining))
            return size_diff

    def Items(self):
        """Get all the items in the bucket."""
        with self._mutex:
            return list(self.items)