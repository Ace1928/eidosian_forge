from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
class WeightedSegment(BaseSegment):
    ndims = 4
    idx, idy = (1, 2)

    @staticmethod
    def get_columns(params):
        return ['edge_id', params.x, params.y, params.weight]

    @staticmethod
    def get_merged_columns(params):
        return ['edge_id', 'src_x', 'src_y', 'dst_x', 'dst_y', params.weight]

    @staticmethod
    @ngjit
    def create_segment(edge):
        return np.array([[edge[0], edge[1], edge[2], edge[5]], [edge[0], edge[3], edge[4], edge[5]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[1] * accuracy), int(point[2] * accuracy)] += point[3]