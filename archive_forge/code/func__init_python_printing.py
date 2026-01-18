from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _init_python_printing(stringify_func, **settings):
    """Setup printing in Python interactive session. """
    import sys
    import builtins

    def _displayhook(arg):
        """Python's pretty-printer display hook.

           This function was adapted from:

            https://www.python.org/dev/peps/pep-0217/

        """
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg, **settings))
            builtins._ = arg
    sys.displayhook = _displayhook