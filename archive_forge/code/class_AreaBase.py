from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
class AreaBase:

    def _plot(self, split_gen, scales, orient):
        patches = defaultdict(list)
        for keys, data, ax in split_gen():
            kws = {}
            data = self._standardize_coordinate_parameters(data, orient)
            resolved = resolve_properties(self, keys, scales)
            verts = self._get_verts(data, orient)
            ax.update_datalim(verts)
            fc = resolve_color(self, keys, '', scales)
            if not resolved['fill']:
                fc = mpl.colors.to_rgba(fc, 0)
            kws['facecolor'] = fc
            kws['edgecolor'] = resolve_color(self, keys, 'edge', scales)
            kws['linewidth'] = resolved['edgewidth']
            kws['linestyle'] = resolved['edgestyle']
            patches[ax].append(mpl.patches.Polygon(verts, **kws))
        for ax, ax_patches in patches.items():
            for patch in ax_patches:
                self._postprocess_artist(patch, ax, orient)
                ax.add_patch(patch)

    def _standardize_coordinate_parameters(self, data, orient):
        return data

    def _postprocess_artist(self, artist, ax, orient):
        pass

    def _get_verts(self, data, orient):
        dv = {'x': 'y', 'y': 'x'}[orient]
        data = data.sort_values(orient, kind='mergesort')
        verts = np.concatenate([data[[orient, f'{dv}min']].to_numpy(), data[[orient, f'{dv}max']].to_numpy()[::-1]])
        if orient == 'y':
            verts = verts[:, ::-1]
        return verts

    def _legend_artist(self, variables, value, scales):
        keys = {v: value for v in variables}
        resolved = resolve_properties(self, keys, scales)
        fc = resolve_color(self, keys, '', scales)
        if not resolved['fill']:
            fc = mpl.colors.to_rgba(fc, 0)
        return mpl.patches.Patch(facecolor=fc, edgecolor=resolve_color(self, keys, 'edge', scales), linewidth=resolved['edgewidth'], linestyle=resolved['edgestyle'], **self.artist_kws)