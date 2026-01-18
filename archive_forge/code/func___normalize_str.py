import numpy as np
import re
from .constants import DRUM_MAP, INSTRUMENT_MAP, INSTRUMENT_CLASSES
def __normalize_str(name):
    """Removes all non-alphanumeric characters from a string and converts
    it to lowercase.

    """
    return ''.join((ch for ch in name if ch.isalnum())).lower()