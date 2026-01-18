import sys
def Fgp(x=None, z=None):
    if x is None:
        return (mnl, matrix(0.0, (n, 1)))
    f = matrix(0.0, (mnl + 1, 1))
    Df = matrix(0.0, (mnl + 1, n))
    blas.copy(g, y)
    base.gemv(F, x, y, beta=1.0)
    if z is not None:
        H = matrix(0.0, (n, n))
    for i, start, stop in ind:
        ymax = max(y[start:stop])
        y[start:stop] = base.exp(y[start:stop] - ymax)
        ysum = blas.asum(y, n=stop - start, offset=start)
        f[i] = ymax + math.log(ysum)
        blas.scal(1.0 / ysum, y, n=stop - start, offset=start)
        base.gemv(F, y, Df, trans='T', m=stop - start, incy=mnl + 1, offsetA=start, offsetx=start, offsety=i)
        if z is not None:
            Fsc[:K[i], :] = F[start:stop, :]
            for k in range(start, stop):
                blas.axpy(Df, Fsc, n=n, alpha=-1.0, incx=mnl + 1, incy=Fsc.size[0], offsetx=i, offsety=k - start)
                blas.scal(math.sqrt(y[k]), Fsc, inc=Fsc.size[0], offset=k - start)
            blas.syrk(Fsc, H, trans='T', k=stop - start, alpha=z[i], beta=1.0)
    if z is None:
        return (f, Df)
    else:
        return (f, Df, H)