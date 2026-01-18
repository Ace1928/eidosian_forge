from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
class GroupedDataFrame(DataFrame):
    """DataFrame grouped by one of several factors."""

    def ungroup(self, *args):
        res = dplyr.ungroup(self, *args)
        return guess_wrap_type(res)(res)