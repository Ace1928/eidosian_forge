import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class Vars(robjects.ListVector):
    """ Aesthetics mapping, using expressions rather than string
    (this is the most common form when using the package in R - it might
    be easier to use AesString when working in Python using rpy2 -
    see class AesString in this Python module).
    """
    _constructor = ggplot2_env['vars']

    @classmethod
    def new(cls, *args):
        """Constructor for the class Vars."""
        new_args = list()
        for a in args:
            new_args.append(rlang.parse_quo(a, env=robjects.baseenv['sys.frame']()))
        res = cls(cls._constructor(*new_args))
        return res