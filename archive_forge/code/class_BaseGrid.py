import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
class BaseGrid(robjects.RObject):

    @classmethod
    def r(cls, *args, **kwargs):
        """ Constructor (as it looks like on the R side)."""
        res = cls._r_constructor(*args, **kwargs)
        return cls(res)