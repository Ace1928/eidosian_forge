import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
class _TestLabeledWrapStruct(_TestWrapStructBase):
    """Test a wrapstruct with value labeling"""

    def test_get_value_label(self):

        class MyHdr(self.header_class):
            _field_recoders = {}
        hdr = MyHdr()
        with pytest.raises(ValueError):
            hdr.get_value_label('improbable')
        assert 'improbable' not in hdr.keys()
        rec = Recoder([[0, 'fullness of heart']], ('code', 'label'))
        hdr._field_recoders['improbable'] = rec
        with pytest.raises(ValueError):
            hdr.get_value_label('improbable')
        for key, value in hdr.items():
            with pytest.raises(ValueError):
                hdr.get_value_label(0)
            if not value.dtype.type in INTEGER_TYPES or not np.isscalar(value):
                continue
            code = int(value)
            rec = Recoder([[code, 'fullness of heart']], ('code', 'label'))
            hdr._field_recoders[key] = rec
            assert hdr.get_value_label(key) == 'fullness of heart'
            new_code = 1 if code == 0 else 0
            hdr[key] = new_code
            assert hdr.get_value_label(key) == f'<unknown code {new_code}>'