def _set_write_buffer_limits(self, high=None, low=None):
    if high is None:
        if low is None:
            high = 64 * 1024
        else:
            high = 4 * low
    if low is None:
        low = high // 4
    if not high >= low >= 0:
        raise ValueError(f'high ({high!r}) must be >= low ({low!r}) must be >= 0')
    self._high_water = high
    self._low_water = low