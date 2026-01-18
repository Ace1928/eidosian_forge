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