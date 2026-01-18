import numpy as np
from ..core import Format, image_as_uint
from ..core.request import RETURN_BYTES
from ._freeimage import FNAME_PER_PLATFORM, IO_FLAGS, download, fi  # noqa
class FreeimageBmpFormat(FreeimageFormat):
    """A BMP format based on the Freeimage library.

    This format supports grayscale, RGB and RGBA images.

    The freeimage plugin requires a `freeimage` binary. If this binary
    not available on the system, it can be downloaded manually from
    <https://github.com/imageio/imageio-binaries> by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for saving
    ---------------------
    compression : bool
        Whether to compress the bitmap using RLE when saving. Default False.
        It seems this does not always work, but who cares, you should use
        PNG anyway.

    """

    class Writer(FreeimageFormat.Writer):

        def _open(self, flags=0, compression=False):
            flags = int(flags)
            if compression:
                flags |= IO_FLAGS.BMP_SAVE_RLE
            else:
                flags |= IO_FLAGS.BMP_DEFAULT
            return FreeimageFormat.Writer._open(self, flags)

        def _append_data(self, im, meta):
            im = image_as_uint(im, bitdepth=8)
            return FreeimageFormat.Writer._append_data(self, im, meta)