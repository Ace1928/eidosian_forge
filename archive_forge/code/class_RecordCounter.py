class RecordCounter:
    """Container for maintains estimates of work requires for fetch.

    Instance of this class is used along with a progress bar to provide
    the user an estimate of the amount of work pending for a fetch (push,
    pull, branch, checkout) operation.
    """

    def __init__(self):
        self.initialized = False
        self.current = 0
        self.key_count = 0
        self.max = 0
        self.STEP = 7

    def is_initialized(self):
        return self.initialized

    def _estimate_max(self, key_count):
        """Estimate the maximum amount of 'inserting stream' work.

        This is just an estimate.
        """
        return int(key_count * 10.3)

    def setup(self, key_count, current=0):
        """Setup RecordCounter with basic estimate of work pending.

        Setup self.max and self.current to reflect the amount of work
        pending for a fetch.
        """
        self.current = current
        self.key_count = key_count
        self.max = self._estimate_max(key_count)
        self.initialized = True

    def increment(self, count):
        """Increment self.current by count.

        Apart from incrementing self.current by count, also ensure
        that self.max > self.current.
        """
        self.current += count
        if self.current > self.max:
            self.max += self.key_count