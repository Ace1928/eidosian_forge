class ConstRunIterator(AbstractRunIterator):
    """Iterate over a constant value without creating a RunList."""

    def __init__(self, length, value):
        self.length = length
        self.end = length
        self.value = value

    def __next__(self):
        yield (0, self.length, self.value)

    def ranges(self, start, end):
        yield (start, end, self.value)

    def __getitem__(self, index):
        return self.value