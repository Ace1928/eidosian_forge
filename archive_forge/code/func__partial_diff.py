from ..libmp.backend import xrange
from .calculus import defun
def _partial_diff(ctx, f, xs, orders, options):
    if not orders:
        return f()
    if not sum(orders):
        return f(*xs)
    i = 0
    for i in range(len(orders)):
        if orders[i]:
            break
    order = orders[i]

    def fdiff_inner(*f_args):

        def inner(t):
            return f(*f_args[:i] + (t,) + f_args[i + 1:])
        return ctx.diff(inner, f_args[i], order, **options)
    orders[i] = 0
    return _partial_diff(ctx, fdiff_inner, xs, orders, options)