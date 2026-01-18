import collections
import numbers
def CastToLinExp(v):
    if isinstance(v, numbers.Number):
        return Constant(v)
    else:
        return v