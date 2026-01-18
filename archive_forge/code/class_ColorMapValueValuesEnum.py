from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ColorMapValueValuesEnum(_messages.Enum):
    """The color scheme used for the highlighted areas. Defaults to
    PINK_GREEN for Integrated Gradients attribution, which shows positive
    attributions in green and negative in pink. Defaults to VIRIDIS for XRAI
    attribution, which highlights the most influential regions in yellow and
    the least influential in blue.

    Values:
      COLOR_MAP_UNSPECIFIED: Should not be used.
      PINK_GREEN: Positive: green. Negative: pink.
      VIRIDIS: Viridis color map: A perceptually uniform color mapping which
        is easier to see by those with colorblindness and progresses from
        yellow to green to blue. Positive: yellow. Negative: blue.
      RED: Positive: red. Negative: red.
      GREEN: Positive: green. Negative: green.
      RED_GREEN: Positive: green. Negative: red.
      PINK_WHITE_GREEN: PiYG palette.
    """
    COLOR_MAP_UNSPECIFIED = 0
    PINK_GREEN = 1
    VIRIDIS = 2
    RED = 3
    GREEN = 4
    RED_GREEN = 5
    PINK_WHITE_GREEN = 6