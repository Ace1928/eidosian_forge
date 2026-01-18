import warnings
import numpy as np
from . import cysoxr
from .cysoxr import QQ, LQ, MQ, HQ, VHQ
from ._version import version as __version__
def _quality_to_enum(q):
    if q in (VHQ, HQ, MQ, LQ, QQ):
        return q
    if type(q) is int:
        raise ValueError(_QUALITY_ERR_STR)
    q = q.lower()
    if q in ('vhq', 'soxr_vhq'):
        return VHQ
    elif q in ('hq', 'soxr_hq'):
        return HQ
    elif q in ('mq', 'soxr_mq'):
        return MQ
    elif q in ('lq', 'soxr_lq'):
        return LQ
    elif q in ('qq', 'soxr_qq'):
        return QQ
    raise ValueError(_QUALITY_ERR_STR)