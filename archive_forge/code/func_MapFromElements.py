import array
import contextlib
import enum
import struct
def MapFromElements(self, elements):
    start = self._StartMap()
    for k, v in elements.items():
        self.Key(k)
        self.Add(v)
    self._EndMap(start)