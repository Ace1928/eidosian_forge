import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def _to_rgba_no_colorcycle(c, alpha=None):
    """
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alpha
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    """
    if isinstance(c, tuple) and len(c) == 2:
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    if alpha is not None and (not 0 <= alpha <= 1):
        raise ValueError("'alpha' must be between 0 and 1, inclusive")
    orig_c = c
    if c is np.ma.masked:
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(c, str):
        if c.lower() == 'none':
            return (0.0, 0.0, 0.0, 0.0)
        try:
            c = _colors_full_map[c]
        except KeyError:
            if len(orig_c) != 1:
                try:
                    c = _colors_full_map[c.lower()]
                except KeyError:
                    pass
    if isinstance(c, str):
        match = re.match('\\A#[a-fA-F0-9]{6}\\Z', c)
        if match:
            return tuple((int(n, 16) / 255 for n in [c[1:3], c[3:5], c[5:7]])) + (alpha if alpha is not None else 1.0,)
        match = re.match('\\A#[a-fA-F0-9]{3}\\Z', c)
        if match:
            return tuple((int(n, 16) / 255 for n in [c[1] * 2, c[2] * 2, c[3] * 2])) + (alpha if alpha is not None else 1.0,)
        match = re.match('\\A#[a-fA-F0-9]{8}\\Z', c)
        if match:
            color = [int(n, 16) / 255 for n in [c[1:3], c[3:5], c[5:7], c[7:9]]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        match = re.match('\\A#[a-fA-F0-9]{4}\\Z', c)
        if match:
            color = [int(n, 16) / 255 for n in [c[1] * 2, c[2] * 2, c[3] * 2, c[4] * 2]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not 0 <= c <= 1:
                raise ValueError(f'Invalid string grayscale value {orig_c!r}. Value must be within 0-1 range')
            return (c, c, c, alpha if alpha is not None else 1.0)
        raise ValueError(f'Invalid RGBA argument: {orig_c!r}')
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    if not np.iterable(c):
        raise ValueError(f'Invalid RGBA argument: {orig_c!r}')
    if len(c) not in [3, 4]:
        raise ValueError('RGBA sequence should have length 3 or 4')
    if not all((isinstance(x, Real) for x in c)):
        raise ValueError(f'Invalid RGBA argument: {orig_c!r}')
    c = tuple(map(float, c))
    if len(c) == 3 and alpha is None:
        alpha = 1
    if alpha is not None:
        c = c[:3] + (alpha,)
    if any((elem < 0 or elem > 1 for elem in c)):
        raise ValueError('RGBA values should be within 0-1 range')
    return c