import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def _draw_mark(mark_type, options={}, axes_options={}, **kwargs):
    """Draw the mark of specified mark type.

    Parameters
    ----------
    mark_type: type
        The type of mark to be drawn
    options: dict (default: {})
        Options for the scales to be created. If a scale labeled 'x' is
        required for that mark, options['x'] contains optional keyword
        arguments for the constructor of the corresponding scale type.
    axes_options: dict (default: {})
        Options for the axes to be created. If an axis labeled 'x' is required
        for that mark, axes_options['x'] contains optional keyword arguments
        for the constructor of the corresponding axis type.
    figure: Figure or None
        The figure to which the mark is to be added.
        If the value is None, the current figure is used.
    cmap: list or string
        List of css colors, or name of bqplot color scheme
    """
    fig = kwargs.pop('figure', current_figure())
    scales = kwargs.pop('scales', {})
    update_context = kwargs.pop('update_context', True)
    cmap = kwargs.pop('cmap', None)
    if cmap is not None:
        options['color'] = dict(options.get('color', {}), **_process_cmap(cmap))
    for name in mark_type.class_trait_names(scaled=True):
        dimension = _get_attribute_dimension(name, mark_type)
        if name not in kwargs:
            continue
        elif name in scales:
            if update_context:
                _context['scales'][dimension] = scales[name]
        elif dimension not in _context['scales']:
            traitlet = mark_type.class_traits()[name]
            rtype = traitlet.get_metadata('rtype')
            dtype = traitlet.validate(None, kwargs[name]).dtype
            compat_scale_types = [Scale.scale_types[key] for key in Scale.scale_types if Scale.scale_types[key].rtype == rtype and issubdtype(dtype, Scale.scale_types[key].dtype)]
            sorted_scales = sorted(compat_scale_types, key=lambda x: x.precedence)
            scales[name] = sorted_scales[-1](**options.get(name, {}))
            if update_context:
                _context['scales'][dimension] = scales[name]
        else:
            scales[name] = _context['scales'][dimension]
    mark = mark_type(scales=scales, **kwargs)
    _context['last_mark'] = mark
    fig.marks = [m for m in fig.marks] + [mark]
    if kwargs.get('axes', True):
        axes(mark, options=axes_options)
    return mark