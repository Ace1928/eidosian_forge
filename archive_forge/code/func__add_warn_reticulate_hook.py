import abc
import os
import typing
import warnings
import weakref
import rpy2.rinterface
import rpy2.rinterface_lib.callbacks
from rpy2.robjects import conversion
def _add_warn_reticulate_hook():
    msg = '\n    WARNING: The R package "reticulate" only fixed recently\n    an issue that caused a segfault when used with rpy2:\n    https://github.com/rstudio/reticulate/pull/1188\n    Make sure that you use a version of that package that includes\n    the fix.\n    '
    rpy2.rinterface.evalr(f'\n    setHook(packageEvent("reticulate", "onLoad"),\n            function(...) cat({repr(msg)}))\n    ')