from cupy import _core
from cupyx.scipy.special._digamma import polevl_definition
from cupyx.scipy.special._gamma import gamma_definition
from cupyx.scipy.special._gammainc import p1evl_definition
Beta and log(abs(beta)) functions.

Also the incomplete beta function and its inverse.

The source code here is an adaptation with minimal changes from the following
files in SciPy's bundled Cephes library:

https://github.com/scipy/scipy/blob/main/scipy/special/cephes/beta.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/incbet.c
https://github.com/scipy/scipy/blob/main/scipy/special/cephes/incbi.c

Cephes Math Library, Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
