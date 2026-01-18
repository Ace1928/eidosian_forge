from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
class UnweightedSegment(BaseSegment):
    ndims = 3
    idx, idy = (1, 2)

    @staticmethod
    def get_columns(params):
        return ['edge_id', params.x, params.y]

    @staticmethod
    def get_merged_columns(params):
        return ['edge_id', 'src_x', 'src_y', 'dst_x', 'dst_y']

    @staticmethod
    @ngjit
    def create_segment(edge):
        return np.array([[edge[0], edge[1], edge[2]], [edge[0], edge[3], edge[4]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[1] * accuracy), int(point[2] * accuracy)] += 1