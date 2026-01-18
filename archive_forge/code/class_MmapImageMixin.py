from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from .. import load as top_load
from ..imageclasses import spatial_axes_first
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..testing import bytesio_round_trip, deprecated_to, expires, memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
class MmapImageMixin:
    """Mixin for testing images that may return memory maps"""
    check_mmap_mode = True

    def get_disk_image(self):
        """Return image, image filename, and flag for required scaling

        Subclasses can do anything to return an image, including loading a
        pre-existing image from disk.

        Returns
        -------
        img : class:`SpatialImage` instance
        fname : str
            Image filename.
        has_scaling : bool
            True if the image array has scaling to apply to the raw image array
            data, False otherwise.
        """
        img_klass = self.image_class
        shape = (3, 4, 2)
        data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
        img = img_klass(data, None)
        fname = 'test' + img_klass.files_types[0][1]
        img.to_filename(fname)
        return (img, fname, False)

    def test_load_mmap(self):
        img_klass = self.image_class
        viral_memmap = memmap_after_ufunc()
        with InTemporaryDirectory():
            img, fname, has_scaling = self.get_disk_image()
            file_map = img.file_map.copy()
            for func, param1 in ((img_klass.from_filename, fname), (img_klass.load, fname), (top_load, fname), (img_klass.from_file_map, file_map)):
                for mmap, expected_mode in ((None, 'c'), (True, 'c'), ('c', 'c'), ('r', 'r'), (False, None)):
                    if has_scaling and (not viral_memmap):
                        expected_mode = None
                    kwargs = {}
                    if mmap is not None:
                        kwargs['mmap'] = mmap
                    back_img = func(param1, **kwargs)
                    back_data = np.asanyarray(back_img.dataobj)
                    if expected_mode is None:
                        assert not isinstance(back_data, np.memmap), f'Should not be a {img_klass.__name__}'
                    else:
                        assert isinstance(back_data, np.memmap), f'Not a {img_klass.__name__}'
                        if self.check_mmap_mode:
                            assert back_data.mode == expected_mode
                    del back_img, back_data
                with pytest.raises(TypeError):
                    func(param1, True)
                with pytest.raises(ValueError):
                    func(param1, mmap='rw')
                with pytest.raises(ValueError):
                    func(param1, mmap='r+')