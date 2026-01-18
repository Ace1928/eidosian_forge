import collections
def _validate_non_negative_integer_or_none(self, value):
    if value is None:
        return
    if not isinstance(value, int):
        raise TypeError('expected value to be an integer')
    if value < 0:
        raise ValueError('expected value to be non-negative', str(value))