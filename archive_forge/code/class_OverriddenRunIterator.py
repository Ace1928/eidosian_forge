class OverriddenRunIterator(AbstractRunIterator):
    """Iterator over a `RunIterator`, with a value temporarily replacing
    a given range.
    """

    def __init__(self, base_iterator, start, end, value):
        """Create a derived iterator.

        :Parameters:
            `start` : int
                Start of range to override
            `end` : int
                End of range to override, exclusive
            `value` : object
                Value to replace over the range

        """
        self.iter = base_iterator
        self.override_start = start
        self.override_end = end
        self.override_value = value

    def ranges(self, start, end):
        if end <= self.override_start or start >= self.override_end:
            for r in self.iter.ranges(start, end):
                yield r
        else:
            if start < self.override_start < end:
                for r in self.iter.ranges(start, self.override_start):
                    yield r
            yield (max(self.override_start, start), min(self.override_end, end), self.override_value)
            if start < self.override_end < end:
                for r in self.iter.ranges(self.override_end, end):
                    yield r

    def __getitem__(self, index):
        if self.override_start <= index < self.override_end:
            return self.override_value
        else:
            return self.iter[index]