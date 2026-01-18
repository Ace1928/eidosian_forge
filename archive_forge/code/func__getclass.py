import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
def _getclass(rclsname):
    if hasattr(rs4classes, rclsname):
        rcls = getattr(rs4classes, rclsname)
    else:
        rcls = type(rclsname, (RS4,), dict())
        setattr(rs4classes, rclsname, rcls)
    return rcls