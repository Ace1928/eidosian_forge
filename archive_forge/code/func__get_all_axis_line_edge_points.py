import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
def _get_all_axis_line_edge_points(self, minmax, maxmin, axis_position=None):
    edgep1s = []
    edgep2s = []
    position = []
    if axis_position in (None, 'default'):
        edgep1, edgep2 = self._get_axis_line_edge_points(minmax, maxmin)
        edgep1s = [edgep1]
        edgep2s = [edgep2]
        position = ['default']
    else:
        edgep1_l, edgep2_l = self._get_axis_line_edge_points(minmax, maxmin, position='lower')
        edgep1_u, edgep2_u = self._get_axis_line_edge_points(minmax, maxmin, position='upper')
        if axis_position in ('lower', 'both'):
            edgep1s.append(edgep1_l)
            edgep2s.append(edgep2_l)
            position.append('lower')
        if axis_position in ('upper', 'both'):
            edgep1s.append(edgep1_u)
            edgep2s.append(edgep2_u)
            position.append('upper')
    return (edgep1s, edgep2s, position)