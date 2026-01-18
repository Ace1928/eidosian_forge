import textwrap
from pprint import PrettyPrinter
from _plotly_utils.utils import *
from _plotly_utils.data_utils import *
class ElidedWrapper(object):
    """
    Helper class that wraps values of certain types and produces a custom
    __repr__() that may be elided and is suitable for use during pretty
    printing
    """

    def __init__(self, v, threshold, indent):
        self.v = v
        self.indent = indent
        self.threshold = threshold

    @staticmethod
    def is_wrappable(v):
        numpy = get_module('numpy')
        if isinstance(v, (list, tuple)) and len(v) > 0 and (not isinstance(v[0], dict)):
            return True
        elif numpy and isinstance(v, numpy.ndarray):
            return True
        elif isinstance(v, str):
            return True
        else:
            return False

    def __repr__(self):
        numpy = get_module('numpy')
        if isinstance(self.v, (list, tuple)):
            res = _list_repr_elided(self.v, threshold=self.threshold, indent=self.indent)
            return res
        elif numpy and isinstance(self.v, numpy.ndarray):
            orig_opts = numpy.get_printoptions()
            numpy.set_printoptions(**dict(orig_opts, threshold=self.threshold, edgeitems=3, linewidth=80))
            res = self.v.__repr__()
            res_lines = res.split('\n')
            res = ('\n' + ' ' * self.indent).join(res_lines)
            numpy.set_printoptions(**orig_opts)
            return res
        elif isinstance(self.v, str):
            if len(self.v) > 80:
                return '(' + repr(self.v[:30]) + ' ... ' + repr(self.v[-30:]) + ')'
            else:
                return self.v.__repr__()
        else:
            return self.v.__repr__()