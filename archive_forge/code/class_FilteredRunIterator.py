class FilteredRunIterator(AbstractRunIterator):
    """Iterate over an `AbstractRunIterator` with filtered values replaced
    by a default value.
    """

    def __init__(self, base_iterator, filter, default):
        """Create a filtered run iterator.

        :Parameters:
            `base_iterator` : `AbstractRunIterator`
                Source of runs.
            `filter` : ``lambda object: bool``
                Function taking a value as parameter, and returning ``True``
                if the value is acceptable, and ``False`` if the default value
                should be substituted.
            `default` : object
                Default value to replace filtered values.

        """
        self.iter = base_iterator
        self.filter = filter
        self.default = default

    def ranges(self, start, end):
        for start, end, value in self.iter.ranges(start, end):
            if self.filter(value):
                yield (start, end, value)
            else:
                yield (start, end, self.default)

    def __getitem__(self, index):
        value = self.iter[index]
        if self.filter(value):
            return value
        return self.default