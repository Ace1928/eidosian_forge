import logging
import numpy as np
from .pillow_legacy import PillowFormat, image_as_uint, ndarray_to_pil
class TIFFFormat(PillowFormat):
    _modes = 'i'
    _description = 'TIFF format (Pillow)'