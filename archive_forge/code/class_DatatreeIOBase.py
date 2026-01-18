from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
class DatatreeIOBase:
    engine: T_NetcdfEngine | None = None

    def test_to_netcdf(self, tmpdir, simple_datatree):
        filepath = tmpdir / 'test.nc'
        original_dt = simple_datatree
        original_dt.to_netcdf(filepath, engine=self.engine)
        roundtrip_dt = open_datatree(filepath, engine=self.engine)
        assert_equal(original_dt, roundtrip_dt)

    def test_netcdf_encoding(self, tmpdir, simple_datatree):
        filepath = tmpdir / 'test.nc'
        original_dt = simple_datatree
        comp = dict(zlib=True, complevel=9)
        enc = {'/set2': {var: comp for var in original_dt['/set2'].ds.data_vars}}
        original_dt.to_netcdf(filepath, encoding=enc, engine=self.engine)
        roundtrip_dt = open_datatree(filepath, engine=self.engine)
        assert roundtrip_dt['/set2/a'].encoding['zlib'] == comp['zlib']
        assert roundtrip_dt['/set2/a'].encoding['complevel'] == comp['complevel']
        enc['/not/a/group'] = {'foo': 'bar'}
        with pytest.raises(ValueError, match='unexpected encoding group.*'):
            original_dt.to_netcdf(filepath, encoding=enc, engine=self.engine)