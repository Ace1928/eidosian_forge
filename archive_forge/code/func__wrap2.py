from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def _wrap2(rfunc, cls, env=robjects.globalenv):

    def func(dataf_a, dataf_b, *args, **kwargs):
        res = rfunc(dataf_a, dataf_b, *args, **kwargs)
        if cls is None:
            return type(dataf_a)(res)
        else:
            return cls(res)
    return func