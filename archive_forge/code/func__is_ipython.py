from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _is_ipython(shell):
    """Is a shell instance an IPython shell?"""
    from sys import modules
    if 'IPython' not in modules:
        return False
    try:
        from IPython.core.interactiveshell import InteractiveShell
    except ImportError:
        try:
            from IPython.iplib import InteractiveShell
        except ImportError:
            return False
    return isinstance(shell, InteractiveShell)