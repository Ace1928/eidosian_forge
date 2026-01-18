import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def _mark_every_path(markevery, tpath, affine, ax):
    """
    Helper function that sorts out how to deal the input
    `markevery` and returns the points where markers should be drawn.

    Takes in the `markevery` value and the line path and returns the
    sub-sampled path.
    """
    codes, verts = (tpath.codes, tpath.vertices)

    def _slice_or_none(in_v, slc):
        """Helper function to cope with `codes` being an ndarray or `None`."""
        if in_v is None:
            return None
        return in_v[slc]
    if isinstance(markevery, Integral):
        markevery = (0, markevery)
    elif isinstance(markevery, Real):
        markevery = (0.0, markevery)
    if isinstance(markevery, tuple):
        if len(markevery) != 2:
            raise ValueError(f'`markevery` is a tuple but its len is not 2; markevery={markevery}')
        start, step = markevery
        if isinstance(step, Integral):
            if not isinstance(start, Integral):
                raise ValueError(f'`markevery` is a tuple with len 2 and second element is an int, but the first element is not an int; markevery={markevery}')
            return Path(verts[slice(start, None, step)], _slice_or_none(codes, slice(start, None, step)))
        elif isinstance(step, Real):
            if not isinstance(start, Real):
                raise ValueError(f'`markevery` is a tuple with len 2 and second element is a float, but the first element is not a float or an int; markevery={markevery}')
            if ax is None:
                raise ValueError('markevery is specified relative to the axes size, but the line does not have a Axes as parent')
            fin = np.isfinite(verts).all(axis=1)
            fverts = verts[fin]
            disp_coords = affine.transform(fverts)
            delta = np.empty((len(disp_coords), 2))
            delta[0, :] = 0
            delta[1:, :] = disp_coords[1:, :] - disp_coords[:-1, :]
            delta = np.hypot(*delta.T).cumsum()
            (x0, y0), (x1, y1) = ax.transAxes.transform([[0, 0], [1, 1]])
            scale = np.hypot(x1 - x0, y1 - y0)
            marker_delta = np.arange(start * scale, delta[-1], step * scale)
            inds = np.abs(delta[np.newaxis, :] - marker_delta[:, np.newaxis])
            inds = inds.argmin(axis=1)
            inds = np.unique(inds)
            return Path(fverts[inds], _slice_or_none(codes, inds))
        else:
            raise ValueError(f'markevery={markevery!r} is a tuple with len 2, but its second element is not an int or a float')
    elif isinstance(markevery, slice):
        return Path(verts[markevery], _slice_or_none(codes, markevery))
    elif np.iterable(markevery):
        try:
            return Path(verts[markevery], _slice_or_none(codes, markevery))
        except (ValueError, IndexError) as err:
            raise ValueError(f'markevery={markevery!r} is iterable but not a valid numpy fancy index') from err
    else:
        raise ValueError(f'markevery={markevery!r} is not a recognized value')