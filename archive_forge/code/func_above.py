from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width
from sympy.utilities.exceptions import sympy_deprecation_warning
def above(self, *args):
    """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
    string, baseline = stringPict.stack(*args + (self,))
    baseline = len(string.splitlines()) - self.height() + self.baseline
    return (string, baseline)