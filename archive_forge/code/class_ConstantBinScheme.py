import math
class ConstantBinScheme(object):

    def __init__(self, bins, min_val, max_val):
        if bins < 2:
            raise ValueError('Must have at least 2 bins.')
        self._min = float(min_val)
        self._max = float(max_val)
        self._bins = int(bins)
        self._bucket_width = (max_val - min_val) / (bins - 2)

    @property
    def bins(self):
        return self._bins

    def from_bin(self, b):
        if b == 0:
            return float('-inf')
        elif b == self._bins - 1:
            return float('inf')
        else:
            return self._min + (b - 1) * self._bucket_width

    def to_bin(self, x):
        if x < self._min:
            return 0
        elif x > self._max:
            return self._bins - 1
        else:
            return int((x - self._min) / self._bucket_width + 1)