import math
def TanimotoDist(ex1, ex2, attrs):
    """
    >>> v1 = [0,1,0,1]
    >>> v2 = [1,0,1,0]
    >>> TanimotoDist(v1,v2,range(4))
    1.0
    >>> v2 = [1,0,1,1]
    >>> TanimotoDist(v1,v2,range(4))
    0.75
    >>> TanimotoDist(v2,v2,range(4))
    0.0

    # this tests Issue 122
    >>> v3 = [0,0,0,0]
    >>> TanimotoDist(v3,v3,range(4))
    1.0

    """
    inter = 0.0
    unin = 0.0
    for i in attrs:
        if ex1[i] or ex2[i]:
            unin += 1
            if ex1[i] and ex2[i]:
                inter += 1
    if unin != 0.0:
        return 1 - inter / unin
    else:
        return 1.0