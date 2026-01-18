from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
class MultiStatistics(dict):
    """Dictionary of :class:`Statistics` object allowing to compute
    statistics on multiple keys using a single call to :meth:`compile`. It
    takes a set of key-value pairs associating a statistics object to a
    unique name. This name can then be used to retrieve the statistics object.

    The following code computes statistics simultaneously on the length and
    the first value of the provided objects.
    ::

        >>> from operator import itemgetter
        >>> import numpy
        >>> len_stats = Statistics(key=len)
        >>> itm0_stats = Statistics(key=itemgetter(0))
        >>> mstats = MultiStatistics(length=len_stats, item=itm0_stats)
        >>> mstats.register("mean", numpy.mean, axis=0)
        >>> mstats.register("max", numpy.max, axis=0)
        >>> mstats.compile([[0.0, 1.0, 1.0, 5.0], [2.0, 5.0]])  # doctest: +SKIP
        {'length': {'mean': 3.0, 'max': 4}, 'item': {'mean': 1.0, 'max': 2.0}}
    """

    def compile(self, data):
        """Calls :meth:`Statistics.compile` with *data* of each
        :class:`Statistics` object.

        :param data: Sequence of objects on which the statistics are computed.
        """
        record = {}
        for name, stats in self.items():
            record[name] = stats.compile(data)
        return record

    @property
    def fields(self):
        return sorted(self.keys())

    def register(self, name, function, *args, **kargs):
        """Register a *function* in each :class:`Statistics` object.

        :param name: The name of the statistics function as it would appear
                     in the dictionary of the statistics object.
        :param function: A function that will compute the desired statistics
                         on the data as preprocessed by the key.
        :param argument: One or more argument (and keyword argument) to pass
                         automatically to the registered function when called,
                         optional.
        """
        for stats in self.values():
            stats.register(name, function, *args, **kargs)