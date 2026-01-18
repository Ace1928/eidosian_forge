from collections import defaultdict
from types import FunctionType
import numpy as np
import pandas as pd
import param
from ..core import Dataset, Dimension, Element2D
from ..core.accessors import Redim
from ..core.operation import Operation
from ..core.util import is_dataframe, max_range, search_indices
from .chart import Points
from .path import Path
from .util import (
class layout_chords(Operation):
    """
    layout_chords computes the locations of each node on a circle and
    the chords connecting them. The amount of radial angle devoted to
    each node and the number of chords are scaled by the value
    dimension of the Chord element. If the values are integers then
    the number of chords is directly scaled by the value, if the
    values are floats then the number of chords are apportioned such
    that the lowest value edge is given one chord and all other nodes
    are given nodes proportional to their weight. The max_chords
    parameter scales the number of chords to be assigned to an edge.

    The chords are computed by interpolating a cubic spline from the
    source to the target node in the graph, the number of samples to
    interpolate the spline with is given by the chord_samples
    parameter.
    """
    chord_samples = param.Integer(default=50, bounds=(0, None), doc='\n        Number of samples per chord for the spline interpolation.')
    max_chords = param.Integer(default=500, doc='\n        Maximum number of chords to render.')

    def _process(self, element, key=None):
        nodes_el = element._nodes
        if nodes_el:
            idx_dim = nodes_el.kdims[-1]
            nodes = nodes_el.dimension_values(idx_dim, expanded=False)
        else:
            source = element.dimension_values(0, expanded=False)
            target = element.dimension_values(1, expanded=False)
            nodes = np.unique(np.concatenate([source, target]))
        max_chords = self.p.max_chords
        src, tgt = (element.dimension_values(i) for i in range(2))
        src_idx = search_indices(src, nodes)
        tgt_idx = search_indices(tgt, nodes)
        if element.vdims:
            values = element.dimension_values(2)
            if values.dtype.kind not in 'uif':
                values = np.ones(len(element), dtype='int')
            else:
                if values.dtype.kind == 'f':
                    values = np.ceil(values * (1.0 / values.min()))
                if values.sum() > max_chords:
                    values = np.ceil(values / float(values.sum()) * max_chords)
                    values = values.astype('int64')
        else:
            values = np.ones(len(element), dtype='int')
        matrix = np.zeros((len(nodes), len(nodes)))
        for s, t, v in zip(src_idx, tgt_idx, values):
            matrix[s, t] += v
        weights_of_areas = matrix.sum(axis=0) + matrix.sum(axis=1)
        areas_in_radians = weights_of_areas / weights_of_areas.sum() * (2 * np.pi)
        points = np.zeros(areas_in_radians.shape[0] + 1)
        points[1:] = areas_in_radians
        points = points.cumsum()
        midpoints = np.convolve(points, [0.5, 0.5], mode='valid')
        mxs = np.cos(midpoints)
        mys = np.sin(midpoints)
        all_areas = []
        for i in range(areas_in_radians.shape[0]):
            n_conn = weights_of_areas[i]
            p0, p1 = (points[i], points[i + 1])
            angles = np.linspace(p0, p1, int(n_conn))
            coords = list(zip(np.cos(angles), np.sin(angles)))
            all_areas.append(coords)
        empty = np.array([[np.nan, np.nan]])
        paths = []
        for i in range(len(element)):
            sidx, tidx = (src_idx[i], tgt_idx[i])
            src_area, tgt_area = (all_areas[sidx], all_areas[tidx])
            n_conns = matrix[sidx, tidx]
            subpaths = []
            for _ in range(int(n_conns)):
                if not src_area or not tgt_area:
                    continue
                x0, y0 = src_area.pop()
                if not tgt_area:
                    continue
                x1, y1 = tgt_area.pop()
                b = quadratic_bezier((x0, y0), (x1, y1), (x0 / 2.0, y0 / 2.0), (x1 / 2.0, y1 / 2.0), steps=self.p.chord_samples)
                subpaths.append(b)
                subpaths.append(empty)
            subpaths = [p for p in subpaths[:-1] if len(p)]
            if subpaths:
                paths.append(np.concatenate(subpaths))
            else:
                paths.append(np.empty((0, 2)))
        if nodes_el:
            if isinstance(nodes_el, Nodes):
                kdims = nodes_el.kdims
            else:
                kdims = Nodes.kdims[:2] + [idx_dim]
            vdims = [vd for vd in nodes_el.vdims if vd not in kdims]
            values = tuple((nodes_el.dimension_values(vd) for vd in vdims))
        else:
            kdims = Nodes.kdims
            values, vdims = ((), [])
        if len(nodes):
            node_data = (mxs, mys, nodes) + values
        else:
            node_data = tuple(([] for _ in kdims + vdims))
        nodes = Nodes(node_data, kdims=kdims, vdims=vdims)
        edges = EdgePaths(paths)
        chord = Chord((element.data, nodes, edges), compute=False)
        chord._angles = points
        return chord