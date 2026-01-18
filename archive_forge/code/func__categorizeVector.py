from fontTools.cffLib import maxStackLimit
def _categorizeVector(v):
    """
    Takes X,Y vector v and returns one of r, h, v, or 0 depending on which
    of X and/or Y are zero, plus tuple of nonzero ones.  If both are zero,
    it returns a single zero still.

    >>> _categorizeVector((0,0))
    ('0', (0,))
    >>> _categorizeVector((1,0))
    ('h', (1,))
    >>> _categorizeVector((0,2))
    ('v', (2,))
    >>> _categorizeVector((1,2))
    ('r', (1, 2))
    """
    if not v[0]:
        if not v[1]:
            return ('0', v[:1])
        else:
            return ('v', v[1:])
    elif not v[1]:
        return ('h', v[:1])
    else:
        return ('r', v)