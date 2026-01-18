from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
from typing import TYPE_CHECKING
class DotBase(Mark):

    def _resolve_paths(self, data):
        paths = []
        path_cache = {}
        marker = data['marker']

        def get_transformed_path(m):
            return m.get_path().transformed(m.get_transform())
        if isinstance(marker, mpl.markers.MarkerStyle):
            return get_transformed_path(marker)
        for m in marker:
            if m not in path_cache:
                path_cache[m] = get_transformed_path(m)
            paths.append(path_cache[m])
        return paths

    def _resolve_properties(self, data, scales):
        resolved = resolve_properties(self, data, scales)
        resolved['path'] = self._resolve_paths(resolved)
        resolved['size'] = resolved['pointsize'] ** 2
        if isinstance(data, dict):
            filled_marker = resolved['marker'].is_filled()
        else:
            filled_marker = [m.is_filled() for m in resolved['marker']]
        resolved['fill'] = resolved['fill'] * filled_marker
        return resolved

    def _plot(self, split_gen, scales, orient):
        for _, data, ax in split_gen():
            offsets = np.column_stack([data['x'], data['y']])
            data = self._resolve_properties(data, scales)
            points = mpl.collections.PathCollection(offsets=offsets, paths=data['path'], sizes=data['size'], facecolors=data['facecolor'], edgecolors=data['edgecolor'], linewidths=data['linewidth'], linestyles=data['edgestyle'], transOffset=ax.transData, transform=mpl.transforms.IdentityTransform(), **self.artist_kws)
            ax.add_collection(points)

    def _legend_artist(self, variables: list[str], value: Any, scales: dict[str, Scale]) -> Artist:
        key = {v: value for v in variables}
        res = self._resolve_properties(key, scales)
        return mpl.collections.PathCollection(paths=[res['path']], sizes=[res['size']], facecolors=[res['facecolor']], edgecolors=[res['edgecolor']], linewidths=[res['linewidth']], linestyles=[res['edgestyle']], transform=mpl.transforms.IdentityTransform(), **self.artist_kws)