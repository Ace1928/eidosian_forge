import numpy as np
from numpy.testing import assert_array_equal
import os
import os.path as osp
import shutil
import tempfile
import h5py as h5
from ..common import ut
from ..._hl.vds import vds_support
class ExcaliburData:
    FEM_PIXELS_PER_CHIP_X = 256
    FEM_PIXELS_PER_CHIP_Y = 256
    FEM_CHIPS_PER_STRIPE_X = 8
    FEM_CHIPS_PER_STRIPE_Y = 1
    FEM_STRIPES_PER_MODULE = 2

    @property
    def sensor_module_dimensions(self):
        x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
        y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y * self.FEM_STRIPES_PER_MODULE
        return (y_pixels, x_pixels)

    @property
    def fem_stripe_dimensions(self):
        x_pixels = self.FEM_PIXELS_PER_CHIP_X * self.FEM_CHIPS_PER_STRIPE_X
        y_pixels = self.FEM_PIXELS_PER_CHIP_Y * self.FEM_CHIPS_PER_STRIPE_Y
        return (y_pixels, x_pixels)

    def generate_sensor_module_image(self, value, dtype='uint16'):
        dset = np.empty(shape=self.sensor_module_dimensions, dtype=dtype)
        dset.fill(value)
        return dset

    def generate_fem_stripe_image(self, value, dtype='uint16'):
        dset = np.empty(shape=self.fem_stripe_dimensions, dtype=dtype)
        dset.fill(value)
        return dset