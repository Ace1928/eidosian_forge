from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
from typing import TYPE_CHECKING
@document_properties
@dataclass
class Bars(BarBase):
    """
    A faster bar mark with defaults more suitable for histograms.

    See also
    --------
    Bar : A bar mark drawn between baseline and data values.

    Examples
    --------
    .. include:: ../docstrings/objects.Bars.rst

    """
    color: MappableColor = Mappable('C0', grouping=False)
    alpha: MappableFloat = Mappable(0.7, grouping=False)
    fill: MappableBool = Mappable(True, grouping=False)
    edgecolor: MappableColor = Mappable(rc='patch.edgecolor', grouping=False)
    edgealpha: MappableFloat = Mappable(1, grouping=False)
    edgewidth: MappableFloat = Mappable(auto=True, grouping=False)
    edgestyle: MappableStyle = Mappable('-', grouping=False)
    width: MappableFloat = Mappable(1, grouping=False)
    baseline: MappableFloat = Mappable(0, grouping=False)

    def _plot(self, split_gen, scales, orient):
        ori_idx = ['x', 'y'].index(orient)
        val_idx = ['y', 'x'].index(orient)
        patches = defaultdict(list)
        for _, data, ax in split_gen():
            bars, _ = self._make_patches(data, scales, orient)
            patches[ax].extend(bars)
        collections = {}
        for ax, ax_patches in patches.items():
            col = mpl.collections.PatchCollection(ax_patches, match_original=True)
            col.sticky_edges[val_idx][:] = (0, np.inf)
            ax.add_collection(col, autolim=False)
            collections[ax] = col
            xys = np.vstack([path.vertices for path in col.get_paths()])
            ax.update_datalim(xys)
        if 'edgewidth' not in scales and isinstance(self.edgewidth, Mappable):
            for ax in collections:
                ax.autoscale_view()

            def get_dimensions(collection):
                edges, widths = ([], [])
                for verts in (path.vertices for path in collection.get_paths()):
                    edges.append(min(verts[:, ori_idx]))
                    widths.append(np.ptp(verts[:, ori_idx]))
                return (np.array(edges), np.array(widths))
            min_width = np.inf
            for ax, col in collections.items():
                edges, widths = get_dimensions(col)
                points = 72 / ax.figure.dpi * abs(ax.transData.transform([edges + widths] * 2) - ax.transData.transform([edges] * 2))
                min_width = min(min_width, min(points[:, ori_idx]))
            linewidth = min(0.1 * min_width, mpl.rcParams['patch.linewidth'])
            for _, col in collections.items():
                col.set_linewidth(linewidth)