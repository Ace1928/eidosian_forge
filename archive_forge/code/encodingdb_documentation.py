import logging
import re
from typing import Dict, Iterable, Optional, cast
from .glyphlist import glyphname2unicode
from .latin_enc import ENCODING
from .psparser import PSLiteral
Unicode values should not be in the range D800 through DFFF because
    that is used for surrogate pairs in UTF-16

    :raises KeyError if unicode digit is invalid
    