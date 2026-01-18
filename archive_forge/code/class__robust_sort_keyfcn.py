class _robust_sort_keyfcn(object):
    """Class for robustly generating sortable keys for arbitrary data.

    Generates keys (for use with Python `sorted()` that are
    (str(type_name), val), where val is the actual value (if the type
    is comparable), otherwise the string representation of the value.
    If str() also fails, we fall back on id().

    This allows sorting lists with mixed types in Python3

    We implement this as a callable object so that we can store the
    user's original key function, if provided

    """
    _typemap = {int: (1, float.__name__), float: (1, float.__name__), str: (1, str.__name__), tuple: (4, tuple.__name__)}

    def __init__(self, key=None):
        self._key = key

    def __call__(self, val):
        """Generate a tuple ( str(type_name), val ) for sorting the value.

        `key=` expects a function.  We are generating a functor so we
        have a convenient place to store the user-provided key and the
        (singleton) _typemap, which maps types to the type-specific
        functions for converting a value to the second argument of the
        sort key.

        """
        if self._key is not None:
            val = self._key(val)
        return self._generate_sort_key(val)

    def _classify_type(self, val):
        _type = val.__class__
        _typename = _type.__name__
        try:
            val < val
            i = 1
            try:
                if bool(val < 1.0) != bool(1.0 < val or 1.0 == val):
                    _typename = float.__name__
            except:
                pass
        except:
            try:
                str(val)
                i = 2
            except:
                i = 3
        self._typemap[_type] = (i, _typename)

    def _generate_sort_key(self, val):
        if val.__class__ not in self._typemap:
            self._classify_type(val)
        i, _typename = self._typemap[val.__class__]
        if i == 1:
            return (_typename, val)
        elif i == 4:
            return (_typename, tuple((self._generate_sort_key(v) for v in val)))
        elif i == 2:
            return (_typename, str(val))
        else:
            return (_typename, id(val))