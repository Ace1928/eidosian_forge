from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
class hammer_bundle(connect_edges):
    """
    Iteratively group edges and return as paths suitable for datashading.

    Breaks each edge into a path with multiple line segments, and
    iteratively curves this path to bundle edges into groups.
    """
    initial_bandwidth = param.Number(default=0.05, bounds=(0.0, None), doc='\n        Initial value of the bandwidth....')
    decay = param.Number(default=0.7, bounds=(0.0, 1.0), doc='\n        Rate of decay in the bandwidth value, with 1.0 indicating no decay.')
    iterations = param.Integer(default=4, bounds=(1, None), doc='\n        Number of passes for the smoothing algorithm')
    batch_size = param.Integer(default=20000, bounds=(1, None), doc='\n        Number of edges to process together')
    tension = param.Number(default=0.3, bounds=(0, None), precedence=-0.5, doc='\n        Exponential smoothing factor to use when smoothing')
    accuracy = param.Integer(default=500, bounds=(1, None), precedence=-0.5, doc='\n        Number of entries in table for...')
    advect_iterations = param.Integer(default=50, bounds=(0, None), precedence=-0.5, doc='\n        Number of iterations to move edges along gradients')
    min_segment_length = param.Number(default=0.008, bounds=(0, None), precedence=-0.5, doc='\n        Minimum length (in data space?) for an edge segment')
    max_segment_length = param.Number(default=0.016, bounds=(0, None), precedence=-0.5, doc='\n        Maximum length (in data space?) for an edge segment')
    weight = param.String(default='weight', allow_None=True, doc='\n        Column name for each edge weight. If None, weights are ignored.')

    def __call__(self, nodes, edges, **params):
        if skimage is None:
            raise ImportError('hammer_bundle operation requires scikit-image. Ensure you install the dependency before applying bundling.')
        p = param.ParamOverrides(self, params)
        xmin, xmax = (np.min(nodes[p.x]), np.max(nodes[p.x]))
        ymin, ymax = (np.min(nodes[p.y]), np.max(nodes[p.y]))
        nodes = nodes.copy()
        nodes[p.x] = minmax_normalize(nodes[p.x], xmin, xmax)
        nodes[p.y] = minmax_normalize(nodes[p.y], ymin, ymax)
        edges, segment_class = _convert_graph_to_edge_segments(nodes, edges, p)
        edge_batches = list(batches(edges, p.batch_size))
        edge_segments = [resample_edges(batch, p.min_segment_length, p.max_segment_length, segment_class.ndims) for batch in edge_batches]
        for i in range(p.iterations):
            bandwidth = p.initial_bandwidth * p.decay ** (i + 1) * p.accuracy
            if bandwidth < 2:
                break
            images = [draw_to_surface(segment, bandwidth, p.accuracy, segment_class.accumulate) for segment in edge_segments]
            overall_image = sum(images)
            gradients = get_gradients(overall_image)
            edge_segments = [advect_resample_all(gradients, segment, p.advect_iterations, p.accuracy, p.min_segment_length, p.max_segment_length, segment_class) for segment in edge_segments]
        edge_segments = [resample_edges(segment, p.min_segment_length, p.max_segment_length, segment_class.ndims) for segment in edge_segments]
        edge_segments = compute(*edge_segments)
        for i in range(10):
            for batch in edge_segments:
                smooth(batch, p.tension, segment_class.idx, segment_class.idy)
        new_segs = []
        for batch in edge_segments:
            new_segs.extend(batch)
        df = _convert_edge_segments_to_dataframe(new_segs, segment_class, p)
        df[p.x] = minmax_denormalize(df[p.x], xmin, xmax)
        df[p.y] = minmax_denormalize(df[p.y], ymin, ymax)
        return df