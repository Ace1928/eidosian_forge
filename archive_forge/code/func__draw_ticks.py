import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _draw_ticks(self, renderer, edgep1, centers, deltas, highs, deltas_per_point, pos):
    ticks = self._update_ticks()
    info = self._axinfo
    index = info['i']
    tickdir = self._get_tickdir(pos)
    tickdelta = deltas[tickdir] if highs[tickdir] else -deltas[tickdir]
    tick_info = info['tick']
    tick_out = tick_info['outward_factor'] * tickdelta
    tick_in = tick_info['inward_factor'] * tickdelta
    tick_lw = tick_info['linewidth']
    edgep1_tickdir = edgep1[tickdir]
    out_tickdir = edgep1_tickdir + tick_out
    in_tickdir = edgep1_tickdir - tick_in
    default_label_offset = 8.0
    points = deltas_per_point * deltas
    for tick in ticks:
        pos = edgep1.copy()
        pos[index] = tick.get_loc()
        pos[tickdir] = out_tickdir
        x1, y1, z1 = proj3d.proj_transform(*pos, self.axes.M)
        pos[tickdir] = in_tickdir
        x2, y2, z2 = proj3d.proj_transform(*pos, self.axes.M)
        labeldeltas = (tick.get_pad() + default_label_offset) * points
        pos[tickdir] = edgep1_tickdir
        pos = _move_from_center(pos, centers, labeldeltas, self._axmask())
        lx, ly, lz = proj3d.proj_transform(*pos, self.axes.M)
        _tick_update_position(tick, (x1, x2), (y1, y2), (lx, ly))
        tick.tick1line.set_linewidth(tick_lw[tick._major])
        tick.draw(renderer)