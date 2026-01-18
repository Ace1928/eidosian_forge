from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _matplotlib_wrapper(o):
    try:
        try:
            return latex_to_png(o, color=forecolor, scale=scale)
        except TypeError:
            return latex_to_png(o)
    except ValueError as e:
        debug('matplotlib exception caught:', repr(e))
        return None