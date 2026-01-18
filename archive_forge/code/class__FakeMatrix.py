import numpy as np
class _FakeMatrix:

    def __init__(self, data):
        self._data = data
        self.__array_interface__ = data.__array_interface__