def _pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
    """
    Generate an or-ed series of shifts for the first 'count' digits.
    Assumes that they fit into 'join_type'.

    (d[2] << 2*n) | (d[1] << 1*n) | d[0]
    """

    def shift(n):
        return ' << (%d * PyLong_SHIFT < 8 * sizeof(%s) ? %d * PyLong_SHIFT : 0)' % (n, join_type, n) if n else ''
    return '(%s)' % ' | '.join(('(((%s)%s[%d])%s)' % (join_type, digits_ptr, i, shift(i)) for i in range(count - 1, -1, -1)))