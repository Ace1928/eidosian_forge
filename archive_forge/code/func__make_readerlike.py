import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def _make_readerlike(stream, byte_order=boc.native_code):

    class R:
        pass
    r = R()
    r.mat_stream = stream
    r.byte_order = byte_order
    r.struct_as_record = True
    r.uint16_codec = sys.getdefaultencoding()
    r.chars_as_strings = False
    r.mat_dtype = False
    r.squeeze_me = False
    return r