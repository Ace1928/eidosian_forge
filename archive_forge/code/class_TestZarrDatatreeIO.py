from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
from xarray.backends.api import open_datatree
from xarray.datatree_.datatree.testing import assert_equal
from xarray.tests import (
@requires_zarr
class TestZarrDatatreeIO:
    engine = 'zarr'

    def test_to_zarr(self, tmpdir, simple_datatree):
        filepath = tmpdir / 'test.zarr'
        original_dt = simple_datatree
        original_dt.to_zarr(filepath)
        roundtrip_dt = open_datatree(filepath, engine='zarr')
        assert_equal(original_dt, roundtrip_dt)

    def test_zarr_encoding(self, tmpdir, simple_datatree):
        import zarr
        filepath = tmpdir / 'test.zarr'
        original_dt = simple_datatree
        comp = {'compressor': zarr.Blosc(cname='zstd', clevel=3, shuffle=2)}
        enc = {'/set2': {var: comp for var in original_dt['/set2'].ds.data_vars}}
        original_dt.to_zarr(filepath, encoding=enc)
        roundtrip_dt = open_datatree(filepath, engine='zarr')
        print(roundtrip_dt['/set2/a'].encoding)
        assert roundtrip_dt['/set2/a'].encoding['compressor'] == comp['compressor']
        enc['/not/a/group'] = {'foo': 'bar'}
        with pytest.raises(ValueError, match='unexpected encoding group.*'):
            original_dt.to_zarr(filepath, encoding=enc, engine='zarr')

    def test_to_zarr_zip_store(self, tmpdir, simple_datatree):
        from zarr.storage import ZipStore
        filepath = tmpdir / 'test.zarr.zip'
        original_dt = simple_datatree
        store = ZipStore(filepath)
        original_dt.to_zarr(store)
        roundtrip_dt = open_datatree(store, engine='zarr')
        assert_equal(original_dt, roundtrip_dt)

    def test_to_zarr_not_consolidated(self, tmpdir, simple_datatree):
        filepath = tmpdir / 'test.zarr'
        zmetadata = filepath / '.zmetadata'
        s1zmetadata = filepath / 'set1' / '.zmetadata'
        filepath = str(filepath)
        original_dt = simple_datatree
        original_dt.to_zarr(filepath, consolidated=False)
        assert not zmetadata.exists()
        assert not s1zmetadata.exists()
        with pytest.warns(RuntimeWarning, match='consolidated'):
            roundtrip_dt = open_datatree(filepath, engine='zarr')
        assert_equal(original_dt, roundtrip_dt)

    def test_to_zarr_default_write_mode(self, tmpdir, simple_datatree):
        import zarr
        simple_datatree.to_zarr(tmpdir)
        with pytest.raises(zarr.errors.ContainsGroupError):
            simple_datatree.to_zarr(tmpdir)