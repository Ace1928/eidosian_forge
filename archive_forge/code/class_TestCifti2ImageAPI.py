import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
class TestCifti2ImageAPI(_TDA, SerializeMixin, DtypeOverrideMixin):
    """Basic validation for Cifti2Image instances"""
    image_maker = ci.Cifti2Image
    header_maker = ci.Cifti2Header
    ni_header_maker = Nifti2Header
    example_shapes = ((2,), (2, 3), (2, 3, 4))
    standard_extension = '.nii'
    storable_dtypes = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64)

    def make_imaker(self, arr, header=None, ni_header=None):
        for idx, sz in enumerate(arr.shape):
            maps = [ci.Cifti2NamedMap(str(value)) for value in range(sz)]
            mim = ci.Cifti2MatrixIndicesMap((idx,), 'CIFTI_INDEX_TYPE_SCALARS', maps=maps)
            header.matrix.append(mim)
        return lambda: self.image_maker(arr.copy(), header, ni_header)