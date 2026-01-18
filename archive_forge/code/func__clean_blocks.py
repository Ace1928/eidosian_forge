import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _clean_blocks(tmp_seq_blocks):
    """Remove curly braces (split codon markers) from the given sequences (PRIVATE)."""
    seq_blocks = []
    for seq_block in tmp_seq_blocks:
        for line_name in seq_block:
            seq_block[line_name] = seq_block[line_name].replace('{', '').replace('}', '')
        seq_blocks.append(seq_block)
    return seq_blocks