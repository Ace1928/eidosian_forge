import numpy as np
def _line_search(self, func, myfprime, xk, pk, gfk, old_fval, old_old_fval, maxstep=0.2, c1=0.23, c2=0.46, xtrapl=1.1, xtrapu=4.0, stpmax=50.0, stpmin=1e-08, args=()):
    self.stpmin = stpmin
    self.pk = pk
    self.stpmax = stpmax
    self.xtrapl = xtrapl
    self.xtrapu = xtrapu
    self.maxstep = maxstep
    phi0 = old_fval
    derphi0 = np.dot(gfk, pk)
    self.dim = len(pk)
    self.gms = np.sqrt(self.dim) * maxstep
    alpha1 = 1.0
    self.no_update = False
    if isinstance(myfprime, type(())):
        fprime = myfprime[0]
        gradient = False
    else:
        fprime = myfprime
        newargs = args
        gradient = True
    fval = old_fval
    gval = gfk
    self.steps = []
    while True:
        stp = self.step(alpha1, phi0, derphi0, c1, c2, self.xtol, self.isave, self.dsave)
        if self.task[:2] == 'FG':
            alpha1 = stp
            fval = func(xk + stp * pk, *args)
            self.fc += 1
            gval = fprime(xk + stp * pk, *newargs)
            if gradient:
                self.gc += 1
            else:
                self.fc += len(xk) + 1
            phi0 = fval
            derphi0 = np.dot(gval, pk)
            self.old_stp = alpha1
            if self.no_update == True:
                break
        else:
            break
    if self.task[:5] == 'ERROR' or self.task[1:4] == 'WARN':
        stp = None
    return (stp, fval, old_fval, self.no_update)