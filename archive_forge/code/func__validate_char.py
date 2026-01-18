import collections
def _validate_char(self, value):
    self._validate_string(value)
    if len(value) != 1:
        raise ValueError('expected value to a string of length one')