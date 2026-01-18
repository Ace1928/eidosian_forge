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