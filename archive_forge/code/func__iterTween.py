from __future__ import division
import math
def _iterTween(startX, startY, endX, endY, intervalSize, tweeningFunc, *args):
    ti = tweeningFunc(0.0, *args)
    yield ((endX - startX) * ti + startX, (endY - startY) * ti + startY)
    n = intervalSize
    while n + 1.1102230246251565e-16 < 1.0:
        ti = tweeningFunc(n, *args)
        yield ((endX - startX) * ti + startX, (endY - startY) * ti + startY)
        n += intervalSize
    ti = tweeningFunc(1.0, *args)
    yield ((endX - startX) * ti + startX, (endY - startY) * ti + startY)