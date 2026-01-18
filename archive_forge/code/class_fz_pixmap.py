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
class fz_pixmap(object):
    """
    Pixmaps represent a set of pixels for a 2 dimensional region of
    a plane. Each pixel has n components per pixel. The components
    are in the order process-components, spot-colors, alpha, where
    there can be 0 of any of those types. The data is in
    premultiplied alpha when rendering, but non-premultiplied for
    colorspace conversions and rescaling.

    x, y: The minimum x and y coord of the region in pixels.

    w, h: The width and height of the region in pixels.

    n: The number of color components in the image.
    	n = num composite colors + num spots + num alphas

    s: The number of spot channels in the image.

    alpha: 0 for no alpha, 1 for alpha present.

    flags: flag bits.
    	Bit 0: If set, draw the image with linear interpolation.
    	Bit 1: If set, free the samples buffer when the pixmap
    	is destroyed.

    stride: The byte offset from the data for any given pixel
    to the data for the same pixel on the row below.

    seps: NULL, or a pointer to a separations structure. If NULL,
    s should be 0.

    xres, yres: Image resolution in dpi. Default is 96 dpi.

    colorspace: Pointer to a colorspace object describing the
    colorspace the pixmap is in. If NULL, the image is a mask.

    samples: Pointer to the first byte of the pixmap sample data.
    This is typically a simple block of memory w * h * n bytes of
    memory in which the components are stored linearly, but with the
    use of appropriate stride values, scanlines can be stored in
    different orders, and have different amounts of padding. The
    first n bytes are components 0 to n-1 for the pixel at (x,y).
    Each successive n bytes gives another pixel in scanline order
    as we move across the line. The start of each scanline is offset
    the start of the previous one by stride bytes.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.fz_pixmap_storable_get, _mupdf.fz_pixmap_storable_set)
    x = property(_mupdf.fz_pixmap_x_get, _mupdf.fz_pixmap_x_set)
    y = property(_mupdf.fz_pixmap_y_get, _mupdf.fz_pixmap_y_set)
    w = property(_mupdf.fz_pixmap_w_get, _mupdf.fz_pixmap_w_set)
    h = property(_mupdf.fz_pixmap_h_get, _mupdf.fz_pixmap_h_set)
    n = property(_mupdf.fz_pixmap_n_get, _mupdf.fz_pixmap_n_set)
    s = property(_mupdf.fz_pixmap_s_get, _mupdf.fz_pixmap_s_set)
    alpha = property(_mupdf.fz_pixmap_alpha_get, _mupdf.fz_pixmap_alpha_set)
    flags = property(_mupdf.fz_pixmap_flags_get, _mupdf.fz_pixmap_flags_set)
    stride = property(_mupdf.fz_pixmap_stride_get, _mupdf.fz_pixmap_stride_set)
    seps = property(_mupdf.fz_pixmap_seps_get, _mupdf.fz_pixmap_seps_set)
    xres = property(_mupdf.fz_pixmap_xres_get, _mupdf.fz_pixmap_xres_set)
    yres = property(_mupdf.fz_pixmap_yres_get, _mupdf.fz_pixmap_yres_set)
    colorspace = property(_mupdf.fz_pixmap_colorspace_get, _mupdf.fz_pixmap_colorspace_set)
    samples = property(_mupdf.fz_pixmap_samples_get, _mupdf.fz_pixmap_samples_set)
    underlying = property(_mupdf.fz_pixmap_underlying_get, _mupdf.fz_pixmap_underlying_set)

    def __init__(self):
        _mupdf.fz_pixmap_swiginit(self, _mupdf.new_fz_pixmap())
    __swig_destroy__ = _mupdf.delete_fz_pixmap