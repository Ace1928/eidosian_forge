import packages is to use `importr()`, for example
import rpy2.robjects as robjects
import rpy2.robjects.constants
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
from rpy2.robjects import rl
import warnings
class ThemeBW(Theme):
    _constructor = ggplot2.theme_bw

    @classmethod
    def new(cls, base_size=12):
        res = cls(cls._constructor(base_size=base_size))
        return res