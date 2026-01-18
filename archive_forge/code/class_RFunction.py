from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
class RFunction(object):
    """Run an R function from Python.

    Parameters
    ----------
    args : str, optional (default: "")
        Comma-separated R argument names and optionally default parameters
    setup : str, optional (default: "")
        R code to run prior to function definition (e.g. loading libraries)
    body : str, optional (default: "")
        R code to run in the body of the function
    cleanup : boolean, optional (default: True)
        If true, clear the R workspace after the function is complete.
        If false, this could result in memory leaks.
    verbose : int, optional (default: 1)
        R script verbosity. For verbose==0, all messages are printed.
        For verbose==1, messages from the function body are printed.
        For verbose==2, messages from the function setup and body are printed.
    """

    def __init__(self, args='', setup='', body='', cleanup=True, verbose=1):
        self.name = 'fun'
        self.args = args
        self.setup = setup
        self.body = body
        self.cleanup = cleanup
        self.verbose = verbose

    @utils._with_pkg(pkg='rpy2', min_version='3.0')
    def _build(self):
        setup_rlang()
        if self.setup != '':
            with _ConsoleWarning(self.verbose - 1):
                rpy2.robjects.r(self.setup)
        function_text = '\n        {name} <- function({args}) {{\n          {body}\n        }}\n        '.format(name=self.name, args=self.args, body=self.body)
        fun = getattr(rpy2.robjects.packages.STAP(function_text, self.name), self.name)
        return fun

    @property
    def function(self):
        try:
            return self._function
        except AttributeError:
            self._function = self._build()
            return self._function

    @utils._with_pkg(pkg='rpy2', min_version='3.0')
    def __call__(self, *args, rpy_cleanup=None, rpy_verbose=None, **kwargs):
        default_verbose = self.verbose
        if rpy_verbose is None:
            rpy_verbose = self.verbose
        else:
            self.verbose = rpy_verbose
        if rpy_cleanup is None:
            rpy_cleanup = self.cleanup
        args = [conversion.py2rpy(a) for a in args]
        kwargs = {k: conversion.py2rpy(v) for k, v in kwargs.items()}
        with _ConsoleWarning(rpy_verbose):
            try:
                robject = self.function(*args, **kwargs)
            except rpy2.rinterface_lib.embedded.RRuntimeError as e:
                try:
                    r_traceback = rpy2.robjects.r("format(rlang::last_trace(), simplify='none', fields=TRUE)")[0]
                except Exception as traceback_exc:
                    r_traceback = f'\n(an error occurred while getting traceback from R){traceback_exc}'
                e.args = (f'{e.args[0]}\n{r_traceback}',)
                raise
            robject = conversion.rpy2py(robject)
            if rpy_cleanup:
                rpy2.robjects.r('rm(list=ls())')
        self.verbose = default_verbose
        return robject