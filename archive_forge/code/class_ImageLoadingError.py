import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
class ImageLoadingError(ValueError):
    """PixBuf returned an error when loading an image.

    The image data is probably corrupted.

    """