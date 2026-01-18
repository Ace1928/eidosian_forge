from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (

    Determine information about a font.

    Args:
        space_width: default space with if no data found
             (normally half the width of a character).
        ft: Font Dictionary

    Returns:
        Font sub-type, space_width criteria(50% of width), encoding, map character-map.
        The font-dictionary itself is suitable for the curious.
    