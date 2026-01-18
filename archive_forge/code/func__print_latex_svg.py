from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _print_latex_svg(o):
    """
        A function that returns a svg rendered by an external latex
        distribution, no fallback available.
        """
    if _can_print(o):
        s = latex(o, mode=latex_mode, **settings)
        if latex_mode == 'plain':
            s = '$\\displaystyle %s$' % s
        try:
            return _svg_wrapper(s)
        except RuntimeError as e:
            debug('preview failed with:', repr(e), ' No fallback available.')