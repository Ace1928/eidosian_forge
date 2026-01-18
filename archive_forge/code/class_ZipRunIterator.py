class ZipRunIterator(AbstractRunIterator):
    """Iterate over multiple run iterators concurrently."""

    def __init__(self, range_iterators):
        self.range_iterators = range_iterators

    def ranges(self, start, end):
        try:
            iterators = [i.ranges(start, end) for i in self.range_iterators]
            starts, ends, values = zip(*[next(i) for i in iterators])
            starts = list(starts)
            ends = list(ends)
            values = list(values)
            while start < end:
                min_end = min(ends)
                yield (start, min_end, values)
                start = min_end
                for i, iterator in enumerate(iterators):
                    if ends[i] == min_end:
                        starts[i], ends[i], values[i] = next(iterator)
        except StopIteration:
            return

    def __getitem__(self, index):
        return [i[index] for i in self.range_iterators]