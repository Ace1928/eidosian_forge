import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _raise_bad_format(format_tag):
    try:
        format_name = WAVE_FORMAT(format_tag).name
    except ValueError:
        format_name = f'{format_tag:#06x}'
    raise ValueError(f'Unknown wave file format: {format_name}. Supported formats: ' + ', '.join((x.name for x in KNOWN_WAVE_FORMATS)))