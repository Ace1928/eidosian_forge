from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _print_latex_matplotlib(o):
    """
        A function that returns a png rendered by mathtext
        """
    if _can_print(o):
        s = latex(o, mode='inline', **settings)
        return _matplotlib_wrapper(s)