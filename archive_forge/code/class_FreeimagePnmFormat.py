import numpy as np
from ..core import Format, image_as_uint
from ..core.request import RETURN_BYTES
from ._freeimage import FNAME_PER_PLATFORM, IO_FLAGS, download, fi  # noqa
class FreeimagePnmFormat(FreeimageFormat):
    """A PNM format based on the Freeimage library.

    This format supports single bit (PBM), grayscale (PGM) and RGB (PPM)
    images, even with ASCII or binary coding.

    The freeimage plugin requires a `freeimage` binary. If this binary
    not available on the system, it can be downloaded manually from
    <https://github.com/imageio/imageio-binaries> by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for saving
    ---------------------
    use_ascii : bool
        Save with ASCII coding. Default True.
    """

    class Writer(FreeimageFormat.Writer):

        def _open(self, flags=0, use_ascii=True):
            flags = int(flags)
            if use_ascii:
                flags |= IO_FLAGS.PNM_SAVE_ASCII
            return FreeimageFormat.Writer._open(self, flags)