import os
import itertools
import numpy as np
import imageio.v3 as iio3
from skimage import data_dir
from skimage.io.collection import ImageCollection, MultiImage, alphanumeric_key
from skimage.io import reset_plugins
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose, fetch
import pytest
class TestImageCollection:
    pics = [fetch('data/brick.png'), fetch('data/color.png'), fetch('data/moon.png')]
    pattern = pics[:2]
    pattern_same_shape = pics[::2]

    def setup_method(self):
        reset_plugins()
        self.images = ImageCollection(self.pattern)
        self.images_matched = ImageCollection(self.pattern_same_shape)
        self.frames_matched = MultiImage(self.pattern_same_shape)

    def test_len(self):
        assert len(self.images) == 2

    def test_getitem(self):
        num = len(self.images)
        for i in range(-num, num):
            assert isinstance(self.images[i], np.ndarray)
        assert_allclose(self.images[0], self.images[-num])

        def return_img(n):
            return self.images[n]
        with testing.raises(IndexError):
            return_img(num)
        with testing.raises(IndexError):
            return_img(-num - 1)

    def test_slicing(self):
        assert type(self.images[:]) is ImageCollection
        assert len(self.images[:]) == 2
        assert len(self.images[:1]) == 1
        assert len(self.images[1:]) == 1
        assert_allclose(self.images[0], self.images[:1][0])
        assert_allclose(self.images[1], self.images[1:][0])
        assert_allclose(self.images[1], self.images[::-1][0])
        assert_allclose(self.images[0], self.images[::-1][1])

    def test_files_property(self):
        assert isinstance(self.images.files, list)

        def set_files(f):
            self.images.files = f
        with testing.raises(AttributeError):
            set_files('newfiles')

    @pytest.mark.skipif(not has_pooch, reason='needs pooch to download data')
    def test_custom_load_func_sequence(self):
        filename = fetch('data/no_time_for_that_tiny.gif')

        def reader(index):
            return iio3.imread(filename, index=index)
        ic = ImageCollection(range(24), load_func=reader)
        assert len(ic) == 24
        assert ic[0].shape == (25, 14, 3)

    @pytest.mark.skipif(not has_pooch, reason='needs pooch to download data')
    def test_custom_load_func_w_kwarg(self):
        load_pattern = fetch('data/no_time_for_that_tiny.gif')

        def load_fn(f, step):
            vid = iio3.imiter(f)
            return list(itertools.islice(vid, None, None, step))
        ic = ImageCollection(load_pattern, load_func=load_fn, step=3)
        assert len(ic) == 1
        assert len(ic[0]) == 8

    def test_custom_load_func(self):

        def load_fn(x):
            return x
        ic = ImageCollection(os.pathsep.join(self.pattern), load_func=load_fn)
        assert_equal(ic[0], self.pattern[0])

    def test_concatenate(self):
        array = self.images_matched.concatenate()
        expected_shape = (len(self.images_matched),) + self.images[0].shape
        assert_equal(array.shape, expected_shape)

    def test_concatenate_mismatched_image_shapes(self):
        with testing.raises(ValueError):
            self.images.concatenate()

    def test_multiimage_imagecollection(self):
        assert_equal(self.images_matched[0], self.frames_matched[0])
        assert_equal(self.images_matched[1], self.frames_matched[1])