from cupy import _core

The source code here is an adaptation with minimal changes from the following
files in SciPy:

https://github.com/scipy/scipy/blob/master/scipy/special/specfun_wrappers.c

Code for psi_spec, gamma2, lpmv0, lpmv was manually translated to C++ from
SciPy's Fortran-77 code located in:
https://github.com/scipy/scipy/blob/master/scipy/special/specfun/specfun.f

The fortran code in scipy originated in the following book.

    "Computation of Special Functions", 1996, John Wiley & Sons, Inc.

    Shanjie Zhang and Jianming Jin

    Copyrighted but permission granted to use code in programs.
