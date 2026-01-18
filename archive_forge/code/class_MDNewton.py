from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
class MDNewton:
    """
    Find the root of a vector function numerically using Newton's method.

    f is a vector function representing a nonlinear equation system.

    x0 is the starting point close to the root.

    J is a function returning the Jacobian matrix for a point.

    Supports overdetermined systems.

    Use the 'norm' keyword to specify which norm to use. Defaults to max-norm.
    The function to calculate the Jacobian matrix can be given using the
    keyword 'J'. Otherwise it will be calculated numerically.

    Please note that this method converges only locally. Especially for high-
    dimensional systems it is not trivial to find a good starting point being
    close enough to the root.

    It is recommended to use a faster, low-precision solver from SciPy [1] or
    OpenOpt [2] to get an initial guess. Afterwards you can use this method for
    root-polishing to any precision.

    [1] http://scipy.org

    [2] http://openopt.org/Welcome
    """
    maxsteps = 10

    def __init__(self, ctx, f, x0, **kwargs):
        self.ctx = ctx
        self.f = f
        if isinstance(x0, (tuple, list)):
            x0 = ctx.matrix(x0)
        assert x0.cols == 1, 'need a vector'
        self.x0 = x0
        if 'J' in kwargs:
            self.J = kwargs['J']
        else:

            def J(*x):
                return ctx.jacobian(f, x)
            self.J = J
        self.norm = kwargs['norm']
        self.verbose = kwargs['verbose']

    def __iter__(self):
        f = self.f
        x0 = self.x0
        norm = self.norm
        J = self.J
        fx = self.ctx.matrix(f(*x0))
        fxnorm = norm(fx)
        cancel = False
        while not cancel:
            fxn = -fx
            Jx = J(*x0)
            s = self.ctx.lu_solve(Jx, fxn)
            if self.verbose:
                print('Jx:')
                print(Jx)
                print('s:', s)
            l = self.ctx.one
            x1 = x0 + s
            while True:
                if x1 == x0:
                    if self.verbose:
                        print("canceled, won't get more excact")
                    cancel = True
                    break
                fx = self.ctx.matrix(f(*x1))
                newnorm = norm(fx)
                if newnorm < fxnorm:
                    fxnorm = newnorm
                    x0 = x1
                    break
                l /= 2
                x1 = x0 + l * s
            yield (x0, fxnorm)