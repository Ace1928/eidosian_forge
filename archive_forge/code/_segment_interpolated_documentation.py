from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from ._colormap import ColorMapKind
from ._interpolated import _InterpolatedGen, interp_lookup

    Gradient colormap by interpolating RGB colors independently

    The input data is the same as Matplotlib's LinearSegmentedColormap
    data.
    