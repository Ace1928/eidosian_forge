def _make_ranges(mydict):
    d = {}
    for key, value in mydict.items():
        d[key] = (value, value)
    return d