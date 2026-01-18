from unittest import mock
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes, GeoAxesSubplot, InterProjectionTransform
class Test_InterProjectionTransform:

    def pc_2_pc(self):
        return InterProjectionTransform(ccrs.PlateCarree(), ccrs.PlateCarree())

    def pc_2_rob(self):
        return InterProjectionTransform(ccrs.PlateCarree(), ccrs.Robinson())

    def rob_2_rob_shifted(self):
        return InterProjectionTransform(ccrs.Robinson(), ccrs.Robinson(central_longitude=0))

    def test_eq(self):
        assert self.pc_2_pc() == self.pc_2_pc()
        assert self.pc_2_rob() == self.pc_2_rob()
        assert self.rob_2_rob_shifted() == self.rob_2_rob_shifted()
        assert not self.pc_2_rob() == self.rob_2_rob_shifted()
        assert not self.pc_2_pc() == 'not a transform obj'

    def test_ne(self):
        assert not self.pc_2_pc() != self.pc_2_pc()
        print(self.pc_2_pc() != self.pc_2_rob())
        assert self.pc_2_pc() != self.pc_2_rob()