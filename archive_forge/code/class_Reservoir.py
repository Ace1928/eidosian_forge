import collections
import random
import threading
class Reservoir:
    """A map-to-arrays container, with deterministic Reservoir Sampling.

    Items are added with an associated key. Items may be retrieved by key, and
    a list of keys can also be retrieved. If size is not zero, then it dictates
    the maximum number of items that will be stored with each key. Once there are
    more items for a given key, they are replaced via reservoir sampling, such
    that each item has an equal probability of being included in the sample.

    Deterministic means that for any given seed and bucket size, the sequence of
    values that are kept for any given tag will always be the same, and that this
    is independent of any insertions on other tags. That is:

    >>> separate_reservoir = reservoir.Reservoir(10)
    >>> interleaved_reservoir = reservoir.Reservoir(10)
    >>> for i in range(100):
    >>>   separate_reservoir.AddItem('key1', i)
    >>> for i in range(100):
    >>>   separate_reservoir.AddItem('key2', i)
    >>> for i in range(100):
    >>>   interleaved_reservoir.AddItem('key1', i)
    >>>   interleaved_reservoir.AddItem('key2', i)

    separate_reservoir and interleaved_reservoir will be in identical states.

    See: https://en.wikipedia.org/wiki/Reservoir_sampling

    Adding items has amortized O(1) runtime.

    Fields:
      always_keep_last: Whether the latest seen sample is always at the
        end of the reservoir. Defaults to True.
      size: An integer of the maximum number of samples.
    """

    def __init__(self, size, seed=0, always_keep_last=True):
        """Creates a new reservoir.

        Args:
          size: The number of values to keep in the reservoir for each tag. If 0,
            all values will be kept.
          seed: The seed of the random number generator to use when sampling.
            Different values for |seed| will produce different samples from the same
            input items.
          always_keep_last: Whether to always keep the latest seen item in the
            end of the reservoir. Defaults to True.

        Raises:
          ValueError: If size is negative or not an integer.
        """
        if size < 0 or size != round(size):
            raise ValueError('size must be nonnegative integer, was %s' % size)
        self._buckets = collections.defaultdict(lambda: _ReservoirBucket(size, random.Random(seed), always_keep_last))
        self._mutex = threading.Lock()
        self.size = size
        self.always_keep_last = always_keep_last

    def Keys(self):
        """Return all the keys in the reservoir.

        Returns:
          ['list', 'of', 'keys'] in the Reservoir.
        """
        with self._mutex:
            return list(self._buckets.keys())

    def Items(self, key):
        """Return items associated with given key.

        Args:
          key: The key for which we are finding associated items.

        Raises:
          KeyError: If the key is not found in the reservoir.

        Returns:
          [list, of, items] associated with that key.
        """
        with self._mutex:
            if key not in self._buckets:
                raise KeyError('Key %s was not found in Reservoir' % key)
            bucket = self._buckets[key]
        return bucket.Items()

    def AddItem(self, key, item, f=lambda x: x):
        """Add a new item to the Reservoir with the given tag.

        If the reservoir has not yet reached full size, the new item is guaranteed
        to be added. If the reservoir is full, then behavior depends on the
        always_keep_last boolean.

        If always_keep_last was set to true, the new item is guaranteed to be added
        to the reservoir, and either the previous last item will be replaced, or
        (with low probability) an older item will be replaced.

        If always_keep_last was set to false, then the new item will replace an
        old item with low probability.

        If f is provided, it will be applied to transform item (lazily, iff item is
          going to be included in the reservoir).

        Args:
          key: The key to store the item under.
          item: The item to add to the reservoir.
          f: An optional function to transform the item prior to addition.
        """
        with self._mutex:
            bucket = self._buckets[key]
        bucket.AddItem(item, f)

    def FilterItems(self, filterFn, key=None):
        """Filter items within a Reservoir, using a filtering function.

        Args:
          filterFn: A function that returns True for the items to be kept.
          key: An optional bucket key to filter. If not specified, will filter all
            all buckets.

        Returns:
          The number of items removed.
        """
        with self._mutex:
            if key:
                if key in self._buckets:
                    return self._buckets[key].FilterItems(filterFn)
                else:
                    return 0
            else:
                return sum((bucket.FilterItems(filterFn) for bucket in self._buckets.values()))