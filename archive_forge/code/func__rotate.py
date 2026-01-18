import numpy as np
from ..core import Format, image_as_uint
from ..core.request import RETURN_BYTES
from ._freeimage import FNAME_PER_PLATFORM, IO_FLAGS, download, fi  # noqa
def _rotate(self, im, meta):
    """Use Orientation information from EXIF meta data to
            orient the image correctly. Freeimage is also supposed to
            support that, and I am pretty sure it once did, but now it
            does not, so let's just do it in Python.
            Edit: and now it works again, just leave in place as a fallback.
            """
    if self.request.kwargs.get('exifrotate', None) == 2:
        try:
            ori = meta['EXIF_MAIN']['Orientation']
        except KeyError:
            pass
        else:
            if ori in [1, 2]:
                pass
            if ori in [3, 4]:
                im = np.rot90(im, 2)
            if ori in [5, 6]:
                im = np.rot90(im, 3)
            if ori in [7, 8]:
                im = np.rot90(im)
            if ori in [2, 4, 5, 7]:
                im = np.fliplr(im)
    return im