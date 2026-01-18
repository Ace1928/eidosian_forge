from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class ScalarHatchProps(HasProps):
    """ Properties relevant to rendering fill regions.

    Mirrors the BokehJS ``properties.Hatch`` class.

    """
    hatch_color = Nullable(Color, default='black', help=_color_help % 'hatching')
    hatch_alpha = Alpha(help=_alpha_help % 'hatching')
    hatch_scale = Size(default=12.0, help=_hatch_scale_help)
    hatch_pattern = Nullable(String, help=_hatch_pattern_help)
    hatch_weight = Size(default=1.0, help=_hatch_weight_help)
    hatch_extra = Dict(String, Instance('bokeh.models.textures.Texture'))