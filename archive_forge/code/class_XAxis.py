import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
class XAxis(Axis):
    _xaxis = xaxis
    _xaxisgrob = grid.xaxisGrob

    @classmethod
    def xaxis(cls, **kwargs):
        """ Constructor (uses the R function grid::xaxis())"""
        res = cls._xaxis(**kwargs)
        return cls(res)

    @classmethod
    def xaxisgrob(cls, **kwargs):
        """ Constructor (uses the R function grid::xaxisgrob())"""
        res = cls._xaxisgrob(**kwargs)
        return cls(res)