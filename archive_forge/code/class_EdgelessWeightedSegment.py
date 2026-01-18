from __future__ import annotations
from math import ceil
from dask import compute, delayed
from pandas import DataFrame
import numpy as np
import pandas as pd
import param
from .utils import ngjit
class EdgelessWeightedSegment(BaseSegment):
    ndims = 3
    idx, idy = (0, 1)

    @staticmethod
    def get_columns(params):
        return [params.x, params.y, params.weight]

    @staticmethod
    def get_merged_columns(params):
        return ['src_x', 'src_y', 'dst_x', 'dst_y', params.weight]

    @staticmethod
    @ngjit
    def create_segment(edge):
        return np.array([[edge[0], edge[1], edge[4]], [edge[2], edge[3], edge[4]]])

    @staticmethod
    @ngjit
    def accumulate(img, point, accuracy):
        img[int(point[0] * accuracy), int(point[1] * accuracy)] += point[2]