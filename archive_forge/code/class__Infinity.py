class _Infinity:
    """Internal type used to represent infinity values."""
    __slots__ = ['_neg']

    def __init__(self, neg):
        self._neg = neg

    def __lt__(self, value):
        if not isinstance(value, _VALID_TYPES):
            return NotImplemented
        return self._neg and (not (isinstance(value, _Infinity) and value._neg))

    def __le__(self, value):
        if not isinstance(value, _VALID_TYPES):
            return NotImplemented
        return self._neg

    def __gt__(self, value):
        if not isinstance(value, _VALID_TYPES):
            return NotImplemented
        return not (self._neg or (isinstance(value, _Infinity) and (not value._neg)))

    def __ge__(self, value):
        if not isinstance(value, _VALID_TYPES):
            return NotImplemented
        return not self._neg

    def __eq__(self, value):
        if not isinstance(value, _VALID_TYPES):
            return NotImplemented
        return isinstance(value, _Infinity) and self._neg == value._neg

    def __ne__(self, value):
        if not isinstance(value, _VALID_TYPES):
            return NotImplemented
        return not isinstance(value, _Infinity) or self._neg != value._neg

    def __repr__(self):
        return 'None'