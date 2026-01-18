import numpy as np
class _FakeMatrix2:

    def __init__(self, data):
        self._data = data

    def __array__(self):
        return self._data