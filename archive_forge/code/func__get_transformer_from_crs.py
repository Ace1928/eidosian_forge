from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
@lru_cache
def _get_transformer_from_crs(src_crs, tgt_crs):
    return Transformer.from_crs(src_crs, tgt_crs, always_xy=True)