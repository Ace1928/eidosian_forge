import logging
import numpy as np
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat
class MngFormat(FreeimageMulti):
    """An Mng format based on the Freeimage library.

    Read only. Seems broken.
    """
    _fif = 6

    def _can_write(self, request):
        return False