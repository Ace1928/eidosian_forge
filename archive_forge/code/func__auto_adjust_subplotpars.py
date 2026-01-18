import numpy as np
import matplotlib as mpl
from matplotlib import _api, artist as martist
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
def _auto_adjust_subplotpars(fig, renderer, shape, span_pairs, subplot_list, ax_bbox_list=None, pad=1.08, h_pad=None, w_pad=None, rect=None):
    """
    Return a dict of subplot parameters to adjust spacing between subplots
    or ``None`` if resulting axes would have zero height or width.

    Note that this function ignores geometry information of subplot itself, but
    uses what is given by the *shape* and *subplot_list* parameters.  Also, the
    results could be incorrect if some subplots have ``adjustable=datalim``.

    Parameters
    ----------
    shape : tuple[int, int]
        Number of rows and columns of the grid.
    span_pairs : list[tuple[slice, slice]]
        List of rowspans and colspans occupied by each subplot.
    subplot_list : list of subplots
        List of subplots that will be used to calculate optimal subplot_params.
    pad : float
        Padding between the figure edge and the edges of subplots, as a
        fraction of the font size.
    h_pad, w_pad : float
        Padding (height/width) between edges of adjacent subplots, as a
        fraction of the font size.  Defaults to *pad*.
    rect : tuple
        (left, bottom, right, top), default: None.
    """
    rows, cols = shape
    font_size_inch = FontProperties(size=mpl.rcParams['font.size']).get_size_in_points() / 72
    pad_inch = pad * font_size_inch
    vpad_inch = h_pad * font_size_inch if h_pad is not None else pad_inch
    hpad_inch = w_pad * font_size_inch if w_pad is not None else pad_inch
    if len(span_pairs) != len(subplot_list) or len(subplot_list) == 0:
        raise ValueError
    if rect is None:
        margin_left = margin_bottom = margin_right = margin_top = None
    else:
        margin_left, margin_bottom, _right, _top = rect
        margin_right = 1 - _right if _right else None
        margin_top = 1 - _top if _top else None
    vspaces = np.zeros((rows + 1, cols))
    hspaces = np.zeros((rows, cols + 1))
    if ax_bbox_list is None:
        ax_bbox_list = [Bbox.union([ax.get_position(original=True) for ax in subplots]) for subplots in subplot_list]
    for subplots, ax_bbox, (rowspan, colspan) in zip(subplot_list, ax_bbox_list, span_pairs):
        if all((not ax.get_visible() for ax in subplots)):
            continue
        bb = []
        for ax in subplots:
            if ax.get_visible():
                bb += [martist._get_tightbbox_for_layout_only(ax, renderer)]
        tight_bbox_raw = Bbox.union(bb)
        tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)
        hspaces[rowspan, colspan.start] += ax_bbox.xmin - tight_bbox.xmin
        hspaces[rowspan, colspan.stop] += tight_bbox.xmax - ax_bbox.xmax
        vspaces[rowspan.start, colspan] += tight_bbox.ymax - ax_bbox.ymax
        vspaces[rowspan.stop, colspan] += ax_bbox.ymin - tight_bbox.ymin
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    if not margin_left:
        margin_left = max(hspaces[:, 0].max(), 0) + pad_inch / fig_width_inch
        suplabel = fig._supylabel
        if suplabel and suplabel.get_in_layout():
            rel_width = fig.transFigure.inverted().transform_bbox(suplabel.get_window_extent(renderer)).width
            margin_left += rel_width + pad_inch / fig_width_inch
    if not margin_right:
        margin_right = max(hspaces[:, -1].max(), 0) + pad_inch / fig_width_inch
    if not margin_top:
        margin_top = max(vspaces[0, :].max(), 0) + pad_inch / fig_height_inch
        if fig._suptitle and fig._suptitle.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(fig._suptitle.get_window_extent(renderer)).height
            margin_top += rel_height + pad_inch / fig_height_inch
    if not margin_bottom:
        margin_bottom = max(vspaces[-1, :].max(), 0) + pad_inch / fig_height_inch
        suplabel = fig._supxlabel
        if suplabel and suplabel.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(suplabel.get_window_extent(renderer)).height
            margin_bottom += rel_height + pad_inch / fig_height_inch
    if margin_left + margin_right >= 1:
        _api.warn_external('Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.')
        return None
    if margin_bottom + margin_top >= 1:
        _api.warn_external('Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.')
        return None
    kwargs = dict(left=margin_left, right=1 - margin_right, bottom=margin_bottom, top=1 - margin_top)
    if cols > 1:
        hspace = hspaces[:, 1:-1].max() + hpad_inch / fig_width_inch
        h_axes = (1 - margin_right - margin_left - hspace * (cols - 1)) / cols
        if h_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout cannot make axes width small enough to accommodate all axes decorations')
            return None
        else:
            kwargs['wspace'] = hspace / h_axes
    if rows > 1:
        vspace = vspaces[1:-1, :].max() + vpad_inch / fig_height_inch
        v_axes = (1 - margin_top - margin_bottom - vspace * (rows - 1)) / rows
        if v_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations.')
            return None
        else:
            kwargs['hspace'] = vspace / v_axes
    return kwargs