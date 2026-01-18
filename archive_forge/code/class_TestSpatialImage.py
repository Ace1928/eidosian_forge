from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
class TestSpatialImage:
    image_class = SpatialImage
    can_save = False

    def test_isolation(self):
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        img = img_klass(arr, aff)
        assert (img.affine == aff).all()
        aff[0, 0] = 99
        assert not np.all(img.affine == aff)
        ihdr = img.header
        img = img_klass(arr, aff, ihdr)
        ihdr.set_zooms((4, 5, 6))
        assert img.header != ihdr

    def test_float_affine(self):
        img_klass = self.image_class
        arr = np.arange(3, dtype=np.int16)
        img = img_klass(arr, np.eye(4, dtype=np.float32))
        assert img.affine.dtype == np.dtype(np.float64)
        img = img_klass(arr, np.eye(4, dtype=np.int16))
        assert img.affine.dtype == np.dtype(np.float64)

    def test_images(self):
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        img = self.image_class(arr, None)
        assert (img.get_fdata() == arr).all()
        assert img.affine is None

    def test_default_header(self):
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        img = self.image_class(arr, None)
        hdr = self.image_class.header_class()
        hdr.set_data_shape(arr.shape)
        hdr.set_data_dtype(arr.dtype)
        assert img.header == hdr

    def test_data_api(self):
        img = self.image_class(DataLike(), None)
        assert (img.get_fdata().flatten() == np.arange(3)).all()
        assert img.shape[:1] == (3,)
        assert np.prod(img.shape) == 3

    def check_dtypes(self, expected, actual):
        assert expected == actual

    def test_data_default(self):
        img_klass = self.image_class
        hdr_klass = self.image_class.header_class
        data = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
        affine = np.eye(4)
        img = img_klass(data, affine)
        self.check_dtypes(data.dtype, img.get_data_dtype())
        header = hdr_klass()
        header.set_data_dtype(np.float32)
        img = img_klass(data, affine, header)
        self.check_dtypes(np.dtype(np.float32), img.get_data_dtype())

    def test_data_shape(self):
        img_klass = self.image_class
        arr = np.arange(4, dtype=np.int16)
        img = img_klass(arr, np.eye(4))
        assert img.shape[:1] == (4,)
        assert np.prod(img.shape) == 4
        img = img_klass(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
        assert img.shape == (2, 3, 4)

    def test_str(self):
        img_klass = self.image_class
        arr = np.arange(5, dtype=np.int16)
        img = img_klass(arr, np.eye(4))
        assert len(str(img)) > 0
        assert img.shape[:1] == (5,)
        assert np.prod(img.shape) == 5
        img = img_klass(np.zeros((2, 3, 4), dtype=np.int16), np.eye(4))
        assert len(str(img)) > 0

    def test_get_fdata(self):
        img_klass = self.image_class
        in_data_template = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        in_data = in_data_template.copy()
        img = img_klass(in_data, None)
        assert in_data is img.dataobj
        assert img.get_fdata(dtype='f4').dtype == np.dtype(np.float32)
        fdata_32 = img.get_fdata(dtype=np.float32)
        assert fdata_32.dtype == np.dtype(np.float32)
        fdata_32[:] = 99
        assert (img.get_fdata(dtype='f4') == 99).all()
        fdata_64 = img.get_fdata()
        assert fdata_64.dtype == np.dtype(np.float64)
        assert (fdata_64 == in_data).all()
        fdata_64[:] = 101
        assert (img.get_fdata(dtype='f8') == 101).all()
        assert (img.get_fdata() == 101).all()
        assert (img.get_fdata(dtype='f4') == in_data).all()
        img.uncache()
        out_data = img.get_fdata()
        assert out_data.dtype == np.dtype(np.float64)
        with pytest.raises(ValueError):
            img.get_fdata(dtype=np.int16)
        with pytest.raises(ValueError):
            img.get_fdata(dtype=np.int32)
        out_data[:] = 42
        assert img.get_fdata() is out_data
        img.uncache()
        assert img.get_fdata() is not out_data
        assert (img.get_fdata() == in_data_template).all()
        if not self.can_save:
            return
        rt_img = bytesio_round_trip(img)
        assert in_data is not rt_img.dataobj
        assert (rt_img.dataobj == in_data).all()
        out_data = rt_img.get_fdata()
        assert (out_data == in_data).all()
        assert rt_img.dataobj is not out_data
        assert out_data.dtype == np.dtype(np.float64)
        assert rt_img.get_fdata() is out_data
        out_data[:] = 42
        rt_img.uncache()
        assert rt_img.get_fdata() is not out_data
        assert (rt_img.get_fdata() == in_data).all()

    @expires('5.0.0')
    def test_get_data(self):
        img_klass = self.image_class
        in_data_template = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        in_data = in_data_template.copy()
        img = img_klass(in_data, None)
        assert in_data is img.dataobj
        with deprecated_to('5.0.0'):
            out_data = img.get_data()
        assert in_data is out_data
        img.uncache()
        assert in_data is out_data
        assert (out_data == in_data_template).all()
        if not self.can_save:
            return
        rt_img = bytesio_round_trip(img)
        assert in_data is not rt_img.dataobj
        assert (rt_img.dataobj == in_data).all()
        with deprecated_to('5.0.0'):
            out_data = rt_img.get_data()
        assert (out_data == in_data).all()
        assert rt_img.dataobj is not out_data
        with deprecated_to('5.0.0'):
            assert rt_img.get_data() is out_data
        out_data[:] = 42
        rt_img.uncache()
        with deprecated_to('5.0.0'):
            assert rt_img.get_data() is not out_data
        with deprecated_to('5.0.0'):
            assert (rt_img.get_data() == in_data).all()

    def test_slicer(self):
        img_klass = self.image_class
        in_data_template = np.arange(240, dtype=np.int16)
        base_affine = np.eye(4)
        t_axis = None
        for dshape in ((4, 5, 6, 2), (8, 5, 6)):
            in_data = in_data_template.copy().reshape(dshape)
            img = img_klass(in_data, base_affine.copy())
            with pytest.raises(TypeError) as exception_manager:
                img[0, 0, 0]
            assert str(exception_manager.value) == 'Cannot slice image objects; consider using `img.slicer[slice]` to generate a sliced image (see documentation for caveats) or slicing image array data with `img.dataobj[slice]` or `img.get_fdata()[slice]`'
            if not spatial_axes_first(img):
                with pytest.raises(ValueError):
                    img.slicer
                continue
            assert hasattr(img.slicer, '__getitem__')
            spatial_zooms = img.header.get_zooms()[:3]
            sliceobj = [slice(None, None, 2)] * 3 + [slice(None)] * (len(dshape) - 3)
            downsampled_img = img.slicer[tuple(sliceobj)]
            assert (downsampled_img.header.get_zooms()[:3] == np.array(spatial_zooms) * 2).all()
            max4d = hasattr(img.header, '_structarr') and 'dims' in img.header._structarr.dtype.fields and (img.header._structarr['dims'].shape == (4,))
            with pytest.raises(IndexError):
                img.slicer[None]
            with pytest.raises(IndexError):
                img.slicer[0]
            with pytest.raises(IndexError):
                img.slicer[:, None]
            with pytest.raises(IndexError):
                img.slicer[:, 0]
            with pytest.raises(IndexError):
                img.slicer[:, :, None]
            with pytest.raises(IndexError):
                img.slicer[:, :, 0]
            if len(img.shape) == 4:
                if max4d:
                    with pytest.raises(ValueError):
                        img.slicer[:, :, :, None]
                else:
                    assert img.slicer[:, :, :, None].shape == img.shape[:3] + (1,) + img.shape[3:]
                assert img.slicer[..., 0].shape == img.shape[:-1]
                assert img.slicer[:, :, :, 0].shape == img.shape[:-1]
            else:
                assert img.slicer[:, :, :, None].shape == img.shape + (1,)
            if len(img.shape) == 3:
                with pytest.raises(IndexError):
                    img.slicer[:, :, :, :, None]
            elif max4d:
                with pytest.raises(ValueError):
                    img.slicer[:, :, :, :, None]
            else:
                assert img.slicer[:, :, :, :, None].shape == img.shape + (1,)
            sliced_i = img.slicer[1:]
            sliced_j = img.slicer[:, 1:]
            sliced_k = img.slicer[:, :, 1:]
            sliced_ijk = img.slicer[1:, 1:, 1:]
            assert (sliced_i.affine[:3, :3] == img.affine[:3, :3]).all()
            assert (sliced_j.affine[:3, :3] == img.affine[:3, :3]).all()
            assert (sliced_k.affine[:3, :3] == img.affine[:3, :3]).all()
            assert (sliced_ijk.affine[:3, :3] == img.affine[:3, :3]).all()
            assert (sliced_i.affine[:, 3] == [1, 0, 0, 1]).all()
            assert (sliced_j.affine[:, 3] == [0, 1, 0, 1]).all()
            assert (sliced_k.affine[:, 3] == [0, 0, 1, 1]).all()
            assert (sliced_ijk.affine[:, 3] == [1, 1, 1, 1]).all()
            assert (img.slicer[:1, :1, :1].affine == img.affine).all()
            with pytest.raises(ValueError):
                img.slicer[:, ::0]
            with pytest.raises(ValueError):
                img.slicer.slice_affine((slice(None), slice(None, None, 0)))
            with pytest.raises(IndexError):
                img.slicer[:0]
            with pytest.raises(IndexError):
                img.slicer[[0]]
            with pytest.raises(IndexError):
                img.slicer[[-1]]
            with pytest.raises(IndexError):
                img.slicer[[0], [-1]]
            slice_elems = np.array((None, Ellipsis, 0, 1, -1, [0], [1], [-1], slice(None), slice(1), slice(-1), slice(1, -1)), dtype=object)
            for n_elems in range(6):
                for _ in range(1 if n_elems == 0 else 10):
                    sliceobj = tuple(np.random.choice(slice_elems, n_elems))
                    try:
                        sliced_img = img.slicer[sliceobj]
                    except (IndexError, ValueError, HeaderDataError):
                        continue
                    sliced_data = in_data[sliceobj]
                    assert np.array_equal(sliced_data, sliced_img.get_fdata())
                    assert np.array_equal(sliced_data, sliced_img.dataobj)
                    assert np.array_equal(sliced_data, img.dataobj[sliceobj])
                    assert np.array_equal(sliced_data, img.get_fdata()[sliceobj])