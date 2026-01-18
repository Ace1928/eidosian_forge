import warnings
from itertools import product
import numpy as np
import pytest
from ..filebasedimages import FileBasedHeader, FileBasedImage, SerializableImage
from .test_image_api import GenericImageAPI, SerializeMixin
class TestFBImageAPI(GenericImageAPI):
    """Validation for FileBasedImage instances"""
    image_maker = FBNumpyImage
    header_maker = FileBasedHeader
    example_shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))
    example_dtypes = (np.int8, np.uint16, np.int32, np.float32)
    can_save = True
    standard_extension = '.npy'

    def make_imaker(self, arr, header=None):
        return lambda: self.image_maker(arr, header)

    def obj_params(self):
        for shape, dtype in product(self.example_shapes, self.example_dtypes):
            arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            hdr = self.header_maker()
            func = self.make_imaker(arr.copy(), hdr)
            params = dict(dtype=dtype, data=arr, shape=shape, is_proxy=False)
            yield (func, params)