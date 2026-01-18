import logging
import numpy as np
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat
class IcoFormat(FreeimageMulti):
    """An ICO format based on the Freeimage library.

    This format supports grayscale, RGB and RGBA images.

    The freeimage plugin requires a `freeimage` binary. If this binary
    is not available on the system, it can be downloaded by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for reading
    ----------------------
    makealpha : bool
        Convert to 32-bit and create an alpha channel from the AND-
        mask when loading. Default False. Note that this returns wrong
        results if the image was already RGBA.

    """
    _fif = 1

    class Reader(FreeimageMulti.Reader):

        def _open(self, flags=0, makealpha=False):
            flags = int(flags)
            if makealpha:
                flags |= IO_FLAGS.ICO_MAKEALPHA
            return FreeimageMulti.Reader._open(self, flags)