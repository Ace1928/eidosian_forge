import itertools
import logging
import numbers
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring, colors, offsetbox
from matplotlib.artist import Artist, allow_rasterization
from matplotlib.cbook import silent_list
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import (Patch, Rectangle, Shadow, FancyBboxPatch,
from matplotlib.collections import (
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo, BboxTransformFrom
from matplotlib.offsetbox import (
from matplotlib.container import ErrorbarContainer, BarContainer, StemContainer
from . import legend_handler
def _parse_legend_args(axs, *args, handles=None, labels=None, **kwargs):
    """
    Get the handles and labels from the calls to either ``figure.legend``
    or ``axes.legend``.

    The parser is a bit involved because we support::

        legend()
        legend(labels)
        legend(handles, labels)
        legend(labels=labels)
        legend(handles=handles)
        legend(handles=handles, labels=labels)

    The behavior for a mixture of positional and keyword handles and labels
    is undefined and issues a warning.

    Parameters
    ----------
    axs : list of `.Axes`
        If handles are not given explicitly, the artists in these Axes are
        used as handles.
    *args : tuple
        Positional parameters passed to ``legend()``.
    handles
        The value of the keyword argument ``legend(handles=...)``, or *None*
        if that keyword argument was not used.
    labels
        The value of the keyword argument ``legend(labels=...)``, or *None*
        if that keyword argument was not used.
    **kwargs
        All other keyword arguments passed to ``legend()``.

    Returns
    -------
    handles : list of (`.Artist` or tuple of `.Artist`)
        The legend handles.
    labels : list of str
        The legend labels.
    kwargs : dict
        *kwargs* with keywords handles and labels removed.

    """
    log = logging.getLogger(__name__)
    handlers = kwargs.get('handler_map')
    if (handles is not None or labels is not None) and args:
        _api.warn_external('You have mixed positional and keyword arguments, some input may be discarded.')
    if handles and labels:
        handles, labels = zip(*zip(handles, labels))
    elif handles is not None and labels is None:
        labels = [handle.get_label() for handle in handles]
    elif labels is not None and handles is None:
        handles = [handle for handle, label in zip(_get_legend_handles(axs, handlers), labels)]
    elif len(args) == 0:
        handles, labels = _get_legend_handles_labels(axs, handlers)
        if not handles:
            log.warning('No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.')
    elif len(args) == 1:
        labels, = args
        if any((isinstance(l, Artist) for l in labels)):
            raise TypeError('A single argument passed to legend() must be a list of labels, but found an Artist in there.')
        handles = [handle for handle, label in zip(_get_legend_handles(axs, handlers), labels)]
    elif len(args) == 2:
        handles, labels = args[:2]
    else:
        raise _api.nargs_error('legend', '0-2', len(args))
    return (handles, labels, kwargs)