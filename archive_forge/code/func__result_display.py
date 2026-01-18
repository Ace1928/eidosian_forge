from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _result_display(self, arg):
    """IPython's pretty-printer display hook, for use in IPython 0.10

           This function was adapted from:

            ipython/IPython/hooks.py:155

        """
    if self.rc.pprint:
        out = stringify_func(arg)
        if '\n' in out:
            print()
        print(out)
    else:
        print(repr(arg))