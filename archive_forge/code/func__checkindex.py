from functools import total_ordering
def _checkindex(self, index):
    length = len(self)
    if 0 <= index < length:
        return index
    if -length <= index < 0:
        return index + length
    raise IndexError('invalid index: %s' % index)