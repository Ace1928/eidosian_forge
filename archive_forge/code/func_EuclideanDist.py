import math
def EuclideanDist(ex1, ex2, attrs):
    """
    >>> v1 = [0,1,0,1]
    >>> v2 = [1,0,1,0]
    >>> EuclideanDist(v1,v2,range(4))
    2.0
    >>> EuclideanDist(v1,v1,range(4))
    0.0
    >>> v2 = [0,0,0,1]
    >>> EuclideanDist(v1,v2,range(4))
    1.0
    >>> v2 = [0,.5,0,.5]
    >>> abs(EuclideanDist(v1,v2,range(4))-1./math.sqrt(2))<1e-4
    1

    """
    dist = 0.0
    for i in attrs:
        dist += (ex1[i] - ex2[i]) ** 2
    dist = math.sqrt(dist)
    return dist