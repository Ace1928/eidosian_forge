from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class fz_image(object):
    """
    Structure is public to allow other structures to
    be derived from it. Do not access members directly.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    key_storable = property(_mupdf.fz_image_key_storable_get, _mupdf.fz_image_key_storable_set)
    w = property(_mupdf.fz_image_w_get, _mupdf.fz_image_w_set)
    h = property(_mupdf.fz_image_h_get, _mupdf.fz_image_h_set)
    n = property(_mupdf.fz_image_n_get, _mupdf.fz_image_n_set)
    bpc = property(_mupdf.fz_image_bpc_get, _mupdf.fz_image_bpc_set)
    imagemask = property(_mupdf.fz_image_imagemask_get, _mupdf.fz_image_imagemask_set)
    interpolate = property(_mupdf.fz_image_interpolate_get, _mupdf.fz_image_interpolate_set)
    use_colorkey = property(_mupdf.fz_image_use_colorkey_get, _mupdf.fz_image_use_colorkey_set)
    use_decode = property(_mupdf.fz_image_use_decode_get, _mupdf.fz_image_use_decode_set)
    decoded = property(_mupdf.fz_image_decoded_get, _mupdf.fz_image_decoded_set)
    scalable = property(_mupdf.fz_image_scalable_get, _mupdf.fz_image_scalable_set)
    orientation = property(_mupdf.fz_image_orientation_get, _mupdf.fz_image_orientation_set)
    mask = property(_mupdf.fz_image_mask_get, _mupdf.fz_image_mask_set)
    xres = property(_mupdf.fz_image_xres_get, _mupdf.fz_image_xres_set)
    yres = property(_mupdf.fz_image_yres_get, _mupdf.fz_image_yres_set)
    colorspace = property(_mupdf.fz_image_colorspace_get, _mupdf.fz_image_colorspace_set)
    drop_image = property(_mupdf.fz_image_drop_image_get, _mupdf.fz_image_drop_image_set)
    get_pixmap = property(_mupdf.fz_image_get_pixmap_get, _mupdf.fz_image_get_pixmap_set)
    get_size = property(_mupdf.fz_image_get_size_get, _mupdf.fz_image_get_size_set)
    colorkey = property(_mupdf.fz_image_colorkey_get, _mupdf.fz_image_colorkey_set)
    decode = property(_mupdf.fz_image_decode_get, _mupdf.fz_image_decode_set)

    def __init__(self):
        _mupdf.fz_image_swiginit(self, _mupdf.new_fz_image())
    __swig_destroy__ = _mupdf.delete_fz_image