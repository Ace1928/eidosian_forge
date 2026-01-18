from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_discrete import scale_discrete
@alias
class scale_colour_manual(scale_color_manual):
    pass