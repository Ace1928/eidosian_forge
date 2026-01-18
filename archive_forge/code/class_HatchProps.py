from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class HatchProps(HasProps):
    """ Properties relevant to rendering fill regions.

    Mirrors the BokehJS ``properties.HatchVector`` class.

    """
    hatch_color = ColorSpec(default='black', help=_color_help % 'hatching')
    hatch_alpha = AlphaSpec(help=_alpha_help % 'hatching')
    hatch_scale = NumberSpec(default=12.0, accept_datetime=False, accept_timedelta=False, help=_hatch_scale_help)
    hatch_pattern = HatchPatternSpec(default=None, help=_hatch_pattern_help)
    hatch_weight = NumberSpec(default=1.0, accept_datetime=False, accept_timedelta=False, help=_hatch_weight_help)
    hatch_extra = Dict(String, Instance('bokeh.models.textures.Texture'))