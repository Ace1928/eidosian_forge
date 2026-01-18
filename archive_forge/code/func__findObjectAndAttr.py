from reportlab.lib.validators import isAnything, DerivedValue
from reportlab.lib.utils import isSeq
from reportlab import rl_config
def _findObjectAndAttr(src, P):
    """Locate the object src.P for P a string, return parent and name of attribute
    """
    P = P.split('.')
    if len(P) == 0:
        return (None, None)
    else:
        for p in P[0:-1]:
            src = getattr(src, p)
        return (src, P[-1])