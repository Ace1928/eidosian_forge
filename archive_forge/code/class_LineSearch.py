import numpy as np
class LineSearch:

    def __init__(self, xtol=1e-14):
        self.xtol = xtol
        self.task = 'START'
        self.isave = np.zeros((2,), np.intc)
        self.dsave = np.zeros((13,), float)
        self.fc = 0
        self.gc = 0
        self.case = 0
        self.old_stp = 0

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

    def step(self, stp, f, g, c1, c2, xtol, isave, dsave):
        if self.task[:5] == 'START':
            if stp < self.stpmin:
                self.task = 'ERROR: STP .LT. minstep'
            if stp > self.stpmax:
                self.task = 'ERROR: STP .GT. maxstep'
            if g >= 0:
                self.task = 'ERROR: INITIAL G >= 0'
            if c1 < 0:
                self.task = 'ERROR: c1 .LT. 0'
            if c2 < 0:
                self.task = 'ERROR: c2 .LT. 0'
            if xtol < 0:
                self.task = 'ERROR: XTOL .LT. 0'
            if self.stpmin < 0:
                self.task = 'ERROR: minstep .LT. 0'
            if self.stpmax < self.stpmin:
                self.task = 'ERROR: maxstep .LT. minstep'
            if self.task[:5] == 'ERROR':
                return stp
            self.bracket = False
            stage = 1
            finit = f
            ginit = g
            gtest = c1 * ginit
            width = self.stpmax - self.stpmin
            width1 = width / 0.5
            stx = 0
            fx = finit
            gx = ginit
            sty = 0
            fy = finit
            gy = ginit
            stmin = 0
            stmax = stp + self.xtrapu * stp
            self.task = 'FG'
            self.save((stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1))
            stp = self.determine_step(stp)
            return stp
        else:
            if self.isave[0] == 1:
                self.bracket = True
            else:
                self.bracket = False
            stage = self.isave[1]
            ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1 = self.dsave
            ftest = finit + stp * gtest
            if stage == 1 and f < ftest and (g >= 0.0):
                stage = 2
            if self.bracket and (stp <= stmin or stp >= stmax):
                self.task = 'WARNING: ROUNDING ERRORS PREVENT PROGRESS'
            if self.bracket and stmax - stmin <= self.xtol * stmax:
                self.task = 'WARNING: XTOL TEST SATISFIED'
            if stp == self.stpmax and f <= ftest and (g <= gtest):
                self.task = 'WARNING: STP = maxstep'
            if stp == self.stpmin and (f > ftest or g >= gtest):
                self.task = 'WARNING: STP = minstep'
            if f <= ftest and abs(g) <= c2 * -ginit:
                self.task = 'CONVERGENCE'
            if self.task[:4] == 'WARN' or self.task[:4] == 'CONV':
                self.save((stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1))
                return stp
            stx, sty, stp, gx, fx, gy, fy = self.update(stx, fx, gx, sty, fy, gy, stp, f, g, stmin, stmax)
            if self.bracket:
                if abs(sty - stx) >= 0.66 * width1:
                    stp = stx + 0.5 * (sty - stx)
                width1 = width
                width = abs(sty - stx)
            if self.bracket:
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else:
                stmin = stp + self.xtrapl * (stp - stx)
                stmax = stp + self.xtrapu * (stp - stx)
            stp = max(stp, self.stpmin)
            stp = min(stp, self.stpmax)
            if stx == stp and stp == self.stpmax and (stmin > self.stpmax):
                self.no_update = True
            if (self.bracket and stp < stmin or stp >= stmax) or (self.bracket and stmax - stmin < self.xtol * stmax):
                stp = stx
            self.task = 'FG'
            self.save((stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1))
            return stp

    def update(self, stx, fx, gx, sty, fy, gy, stp, fp, gp, stpmin, stpmax):
        sign = gp * (gx / abs(gx))
        if fp > fx:
            self.case = 1
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt((theta / s) ** 2.0 - gx / s * (gp / s))
            if stp < stx:
                gamma = -gamma
            p = gamma - gx + theta
            q = gamma - gx + gamma + gp
            r = p / q
            stpc = stx + r * (stp - stx)
            stpq = stx + gx / ((fx - fp) / (stp - stx) + gx) / 2.0 * (stp - stx)
            if abs(stpc - stx) < abs(stpq - stx):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2.0
            self.bracket = True
        elif sign < 0:
            self.case = 2
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt((theta / s) ** 2 - gx / s * (gp / s))
            if stp > stx:
                gamma = -gamma
            p = gamma - gp + theta
            q = gamma - gp + gamma + gx
            r = p / q
            stpc = stp + r * (stx - stp)
            stpq = stp + gp / (gp - gx) * (stx - stp)
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            self.bracket = True
        elif abs(gp) < abs(gx):
            self.case = 3
            theta = 3.0 * (fx - fp) / (stp - stx) + gx + gp
            s = max(abs(theta), abs(gx), abs(gp))
            gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - gx / s * (gp / s)))
            if stp > stx:
                gamma = -gamma
            p = gamma - gp + theta
            q = gamma + (gx - gp) + gamma
            r = p / q
            if r < 0.0 and gamma != 0:
                stpc = stp + r * (stx - stp)
            elif stp > stx:
                stpc = stpmax
            else:
                stpc = stpmin
            stpq = stp + gp / (gp - gx) * (stx - stp)
            if self.bracket:
                if abs(stpc - stp) < abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                if stp > stx:
                    stpf = min(stp + 0.66 * (sty - stp), stpf)
                else:
                    stpf = max(stp + 0.66 * (sty - stp), stpf)
            else:
                if abs(stpc - stp) > abs(stpq - stp):
                    stpf = stpc
                else:
                    stpf = stpq
                stpf = min(stpmax, stpf)
                stpf = max(stpmin, stpf)
        else:
            self.case = 4
            if self.bracket:
                theta = 3.0 * (fp - fy) / (sty - stp) + gy + gp
                s = max(abs(theta), abs(gy), abs(gp))
                gamma = s * np.sqrt((theta / s) ** 2 - gy / s * (gp / s))
                if stp > sty:
                    gamma = -gamma
                p = gamma - gp + theta
                q = gamma - gp + gamma + gy
                r = p / q
                stpc = stp + r * (sty - stp)
                stpf = stpc
            elif stp > stx:
                stpf = stpmax
            else:
                stpf = stpmin
        if fp > fx:
            sty = stp
            fy = fp
            gy = gp
        else:
            if sign < 0:
                sty = stx
                fy = fx
                gy = gx
            stx = stp
            fx = fp
            gx = gp
        stp = self.determine_step(stpf)
        return (stx, sty, stp, gx, fx, gy, fy)

    def determine_step(self, stp):
        dr = stp - self.old_stp
        x = np.reshape(self.pk, (-1, 3))
        steplengths = ((dr * x) ** 2).sum(1) ** 0.5
        maxsteplength = pymax(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength
        stp = self.old_stp + dr
        return stp

    def save(self, data):
        if self.bracket:
            self.isave[0] = 1
        else:
            self.isave[0] = 0
        self.isave[1] = data[0]
        self.dsave = data[1:]