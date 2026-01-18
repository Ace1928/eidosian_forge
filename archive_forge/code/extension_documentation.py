import re
from distutils.extension import Extension as old_Extension

    Parameters
    ----------
    name : str
        Extension name.
    sources : list of str
        List of source file locations relative to the top directory of
        the package.
    extra_compile_args : list of str
        Extra command line arguments to pass to the compiler.
    extra_f77_compile_args : list of str
        Extra command line arguments to pass to the fortran77 compiler.
    extra_f90_compile_args : list of str
        Extra command line arguments to pass to the fortran90 compiler.
    