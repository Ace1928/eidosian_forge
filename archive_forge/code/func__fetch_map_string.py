import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from packaging.version import Version
from matplotlib import cm
from matplotlib import colors
from branca.colormap import StepColormap
def _fetch_map_string(self, m):
    out = m._parent.render()
    out_str = ''.join(out.split())
    return out_str