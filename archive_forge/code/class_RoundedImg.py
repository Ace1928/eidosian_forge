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
class RoundedImg(cimg_nest.Img):

    @staticmethod
    def world_file_extent(*args, **kwargs):
        """
        Takes account for the fact that the image tiles are stored with
        imprecise tfw files.

        """
        extent, pix_size = cimg_nest.Img.world_file_extent(*args, **kwargs)
        extent = tuple((round(v, 4) for v in extent))
        pix_size = tuple((round(v, 4) for v in pix_size))
        return (extent, pix_size)