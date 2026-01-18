from ...data import sampler
class IntervalSampler(sampler.Sampler):
    """Samples elements from [0, length) at fixed intervals.

    Parameters
    ----------
    length : int
        Length of the sequence.
    interval : int
        The number of items to skip between two samples.
    rollover : bool, default True
        Whether to start again from the first skipped item after reaching the end.
        If true, this sampler would start again from the first skipped item until all items
        are visited.
        Otherwise, iteration stops when end is reached and skipped items are ignored.

    Examples
    --------
    >>> sampler = contrib.data.IntervalSampler(13, interval=3)
    >>> list(sampler)
    [0, 3, 6, 9, 12, 1, 4, 7, 10, 2, 5, 8, 11]
    >>> sampler = contrib.data.IntervalSampler(13, interval=3, rollover=False)
    >>> list(sampler)
    [0, 3, 6, 9, 12]
    """

    def __init__(self, length, interval, rollover=True):
        assert interval <= length, 'Interval {} must be smaller than or equal to length {}'.format(interval, length)
        self._length = length
        self._interval = interval
        self._rollover = rollover

    def __iter__(self):
        for i in range(self._interval if self._rollover else 1):
            for j in range(i, self._length, self._interval):
                yield j

    def __len__(self):
        return self._length