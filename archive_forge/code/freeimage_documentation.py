import numpy as np
from ..core import Format, image_as_uint
from ..core.request import RETURN_BYTES
from ._freeimage import FNAME_PER_PLATFORM, IO_FLAGS, download, fi  # noqa
Use Orientation information from EXIF meta data to
            orient the image correctly. Freeimage is also supposed to
            support that, and I am pretty sure it once did, but now it
            does not, so let's just do it in Python.
            Edit: and now it works again, just leave in place as a fallback.
            