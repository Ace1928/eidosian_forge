def _quick(a, l, r):
    count = 0
    if l < r:
        s, count = _partition(a, l, r)
        count += _quick(a, l, s - 1)
        count += _quick(a, s + 1, r)
    return count