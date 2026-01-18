import collections
def _validate_string(self, value):
    if not isinstance(value, (str, ''.__class__)):
        raise TypeError('expected value to be a string')