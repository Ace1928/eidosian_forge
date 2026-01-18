from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
def _make_pipe2(rfunc, cls, env=robjects.globalenv):
    """
    :param rfunc: An R function.
    :param cls: The class to use wrap the result of `rfunc`.
    :param env: A R environment.
    :rtype: A function."""

    def inner(obj_a, obj_b, *args, **kwargs):
        res = rfunc(obj_a, obj_b, *args, **kwargs)
        return cls(res)
    return inner