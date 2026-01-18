import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class GBaseObject(robjects.Environment):

    @classmethod
    def new(cls, *args, **kwargs):
        args_list = list(args)
        res = cls(cls._constructor(*args_list, **kwargs))
        return res