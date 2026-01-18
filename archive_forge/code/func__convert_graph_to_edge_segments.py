from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
def _convert_graph_to_edge_segments(nodes, edges, params):
    """
    Merge graph dataframes into a list of edge segments.

    Given a graph defined as a pair of dataframes (nodes and edges), the
    nodes (id, coordinates) and edges (id, source, target, weight) are
    joined by node id to create a single dataframe with each source/target
    of an edge (including its optional weight) replaced with the respective
    coordinates. For both nodes and edges, each id column is assumed to be
    the index.

    We also return the dimensions of each point in the final dataframe and
    the accumulator function for drawing to an image.
    """
    df = pd.merge(edges, nodes, left_on=[params.source], right_index=True)
    df = df.rename(columns={params.x: 'src_x', params.y: 'src_y'})
    df = pd.merge(df, nodes, left_on=[params.target], right_index=True)
    df = df.rename(columns={params.x: 'dst_x', params.y: 'dst_y'})
    df = df.sort_index()
    df = df.reset_index()
    include_edge_id = params.include_edge_id
    if include_edge_id:
        df = df.rename(columns={'id': 'edge_id'})
    include_weight = params.weight and params.weight in edges
    if include_edge_id:
        if include_weight:
            segment_class = WeightedSegment
        else:
            segment_class = UnweightedSegment
    elif include_weight:
        segment_class = EdgelessWeightedSegment
    else:
        segment_class = EdgelessUnweightedSegment
    df = df.filter(items=segment_class.get_merged_columns(params))
    edge_segments = []
    for tup in df.itertuples():
        edge = (tup.src_x, tup.src_y, tup.dst_x, tup.dst_y)
        if include_edge_id:
            edge = (tup.edge_id,) + edge
        if include_weight:
            edge += (getattr(tup, params.weight),)
        edge_segments.append(segment_class.create_segment(edge))
    return (edge_segments, segment_class)