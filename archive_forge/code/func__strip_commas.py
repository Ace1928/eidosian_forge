from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
@classmethod
def _strip_commas(cls, kw):
    """Strip out any leading/training commas from the token"""
    kw = kw[:-1] if kw[-1] == ',' else kw
    return kw[1:] if kw[0] == ',' else kw