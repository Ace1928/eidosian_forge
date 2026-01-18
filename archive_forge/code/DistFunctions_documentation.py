import math

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

    