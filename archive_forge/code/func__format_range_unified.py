import email.parser
import time
from difflib import SequenceMatcher
from typing import BinaryIO, Optional, TextIO, Union
from .objects import S_ISGITLINK, Blob, Commit
from .pack import ObjectContainer
def _format_range_unified(start, stop):
    """Convert range to the "ed" format."""
    beginning = start + 1
    length = stop - start
    if length == 1:
        return f'{beginning}'
    if not length:
        beginning -= 1
    return f'{beginning},{length}'