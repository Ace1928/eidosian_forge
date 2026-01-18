from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
from seaborn._marks.base import (
def _postprocess_artist(self, artist, ax, orient):
    artist.set_linewidth(artist.get_linewidth() * 2)
    linestyle = artist.get_linestyle()
    if linestyle[1]:
        linestyle = (linestyle[0], tuple((x / 2 for x in linestyle[1])))
    artist.set_linestyle(linestyle)
    artist.set_clip_path(artist.get_path(), artist.get_transform() + ax.transData)
    if self.artist_kws.get('clip_on', True):
        artist.set_clip_box(ax.bbox)
    val_idx = ['y', 'x'].index(orient)
    artist.sticky_edges[val_idx][:] = (0, np.inf)