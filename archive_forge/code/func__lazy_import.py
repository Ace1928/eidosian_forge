import importlib
import importlib.util
import inspect
import os
import sys
import types
def _lazy_import(fullname):
    """Return a lazily imported proxy for a module or library.

    Warning
    -------
    Importing using this function can currently cause trouble
    when the user tries to import from a subpackage of a module before
    the package is fully imported. In particular, this idiom may not work:

      np = lazy_import("numpy")
      from numpy.lib import recfunctions

    This is due to a difference in the way Python's LazyLoader handles
    subpackage imports compared to the normal import process. Hopefully
    we will get Python's LazyLoader to fix this, or find a workaround.
    In the meantime, this is a potential problem.

    The workaround is to import numpy before importing from the subpackage.

    Notes
    -----
    We often see the following pattern::

      def myfunc():
          import scipy as sp
          sp.argmin(...)
          ....

    This is to prevent a library, in this case `scipy`, from being
    imported at function definition time, since that can be slow.

    This function provides a proxy module that, upon access, imports
    the actual module.  So the idiom equivalent to the above example is::

      sp = lazy.load("scipy")

      def myfunc():
          sp.argmin(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of the library.

    Parameters
    ----------
    fullname : str
        The full name of the package or subpackage to import.  For example::

          sp = lazy.load('scipy')  # import scipy as sp
          spla = lazy.load('scipy.linalg')  # import scipy.linalg as spla

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module. Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    try:
        return sys.modules[fullname]
    except:
        pass
    spec = importlib.util.find_spec(fullname)
    if spec is None:
        try:
            parent = inspect.stack()[1]
            frame_data = {'spec': fullname, 'filename': parent.filename, 'lineno': parent.lineno, 'function': parent.function, 'code_context': parent.code_context}
            return DelayedImportErrorModule(frame_data, 'DelayedImportErrorModule')
        finally:
            del parent
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    loader = importlib.util.LazyLoader(spec.loader)
    loader.exec_module(module)
    return module