import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def _fill_and_stroke(self, ctx, fill_c, alpha, alpha_overrides):
    if fill_c is not None:
        ctx.save()
        if len(fill_c) == 3 or alpha_overrides:
            ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], alpha)
        else:
            ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], fill_c[3])
        ctx.fill_preserve()
        ctx.restore()
    ctx.stroke()