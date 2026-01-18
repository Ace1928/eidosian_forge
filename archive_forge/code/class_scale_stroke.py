from warnings import warn
import numpy as np
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
@alias
class scale_stroke(scale_stroke_continuous):
    pass