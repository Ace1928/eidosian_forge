import builtins
from itertools import islice
from _thread import get_ident
def _repr_iterable(self, x, level, left, right, maxiter, trail=''):
    n = len(x)
    if level <= 0 and n:
        s = self.fillvalue
    else:
        newlevel = level - 1
        repr1 = self.repr1
        pieces = [repr1(elem, newlevel) for elem in islice(x, maxiter)]
        if n > maxiter:
            pieces.append(self.fillvalue)
        s = ', '.join(pieces)
        if n == 1 and trail:
            right = trail + right
    return '%s%s%s' % (left, s, right)