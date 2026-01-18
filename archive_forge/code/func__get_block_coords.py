import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _get_block_coords(parsed_seq, row_dict, has_ner=False):
    """Return a list of start, end coordinates for each given block in the sequence (PRIVATE)."""
    start = 0
    coords = []
    if not has_ner:
        splitter = _RE_EXON
    else:
        splitter = _RE_NER
    seq = parsed_seq[row_dict['query']]
    for block in re.split(splitter, seq):
        start += seq[start:].find(block)
        end = start + len(block)
        coords.append((start, end))
    return coords