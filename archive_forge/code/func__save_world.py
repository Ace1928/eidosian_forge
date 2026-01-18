import io
from pathlib import Path
import pickle
import shutil
import sys
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from PIL import Image
import pytest
import shapely.geometry as sgeom
from cartopy import config
import cartopy.io.img_nest as cimg_nest
import cartopy.io.img_tiles as cimgt
def _save_world(fname, args):
    _world = '{x_pix_size}\n{y_rotation}\n{x_rotation}\n{y_pix_size}\n{x_center}\n{y_center}\n'
    with open(fname, 'w') as fh:
        fh.write(_world.format(**args))