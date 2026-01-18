from fontTools.misc.transform import Identity, Scale
from math import atan2, ceil, cos, fabs, isfinite, pi, radians, sin, sqrt, tan
Convert SVG Path's elliptical arcs to Bezier curves.

The code is mostly adapted from Blink's SVGPathNormalizer::DecomposeArcToCubic
https://github.com/chromium/chromium/blob/93831f2/third_party/
blink/renderer/core/svg/svg_path_parser.cc#L169-L278
