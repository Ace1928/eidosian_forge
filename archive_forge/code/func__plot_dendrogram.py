import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def _plot_dendrogram(icoords, dcoords, ivl, p, n, mh, orientation, no_labels, color_list, leaf_font_size=None, leaf_rotation=None, contraction_marks=None, ax=None, above_threshold_color='C0'):
    try:
        if ax is None:
            import matplotlib.pylab
        import matplotlib.patches
        import matplotlib.collections
    except ImportError as e:
        raise ImportError('You must install the matplotlib library to plot the dendrogram. Use no_plot=True to calculate the dendrogram without plotting.') from e
    if ax is None:
        ax = matplotlib.pylab.gca()
        trigger_redraw = True
    else:
        trigger_redraw = False
    ivw = len(ivl) * 10
    dvw = mh + mh * 0.05
    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)
    if orientation in ('top', 'bottom'):
        if orientation == 'top':
            ax.set_ylim([0, dvw])
            ax.set_xlim([0, ivw])
        else:
            ax.set_ylim([dvw, 0])
            ax.set_xlim([0, ivw])
        xlines = icoords
        ylines = dcoords
        if no_labels:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(iv_ticks)
            if orientation == 'top':
                ax.xaxis.set_ticks_position('bottom')
            else:
                ax.xaxis.set_ticks_position('top')
            for line in ax.get_xticklines():
                line.set_visible(False)
            leaf_rot = float(_get_tick_rotation(len(ivl))) if leaf_rotation is None else leaf_rotation
            leaf_font = float(_get_tick_text_size(len(ivl))) if leaf_font_size is None else leaf_font_size
            ax.set_xticklabels(ivl, rotation=leaf_rot, size=leaf_font)
    elif orientation in ('left', 'right'):
        if orientation == 'left':
            ax.set_xlim([dvw, 0])
            ax.set_ylim([0, ivw])
        else:
            ax.set_xlim([0, dvw])
            ax.set_ylim([0, ivw])
        xlines = dcoords
        ylines = icoords
        if no_labels:
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_yticks(iv_ticks)
            if orientation == 'left':
                ax.yaxis.set_ticks_position('right')
            else:
                ax.yaxis.set_ticks_position('left')
            for line in ax.get_yticklines():
                line.set_visible(False)
            leaf_font = float(_get_tick_text_size(len(ivl))) if leaf_font_size is None else leaf_font_size
            if leaf_rotation is not None:
                ax.set_yticklabels(ivl, rotation=leaf_rotation, size=leaf_font)
            else:
                ax.set_yticklabels(ivl, size=leaf_font)
    colors_used = _remove_dups(color_list)
    color_to_lines = {}
    for color in colors_used:
        color_to_lines[color] = []
    for xline, yline, color in zip(xlines, ylines, color_list):
        color_to_lines[color].append(list(zip(xline, yline)))
    colors_to_collections = {}
    for color in colors_used:
        coll = matplotlib.collections.LineCollection(color_to_lines[color], colors=(color,))
        colors_to_collections[color] = coll
    for color in colors_used:
        if color != above_threshold_color:
            ax.add_collection(colors_to_collections[color])
    if above_threshold_color in colors_to_collections:
        ax.add_collection(colors_to_collections[above_threshold_color])
    if contraction_marks is not None:
        Ellipse = matplotlib.patches.Ellipse
        for x, y in contraction_marks:
            if orientation in ('left', 'right'):
                e = Ellipse((y, x), width=dvw / 100, height=1.0)
            else:
                e = Ellipse((x, y), width=1.0, height=dvw / 100)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor('k')
    if trigger_redraw:
        matplotlib.pylab.draw_if_interactive()