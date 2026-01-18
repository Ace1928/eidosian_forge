import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
def _palette_is_grayscale(pil_image):
    if pil_image.mode != 'P':
        return False
    elif pil_image.info.get('transparency', None):
        return False
    palette = np.asarray(pil_image.getpalette()).reshape((-1, 3))
    start, stop = pil_image.getextrema()
    valid_palette = palette[start:stop + 1]
    return np.allclose(np.diff(valid_palette), 0)