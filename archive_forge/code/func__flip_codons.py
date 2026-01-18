import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _flip_codons(codon_seq, target_seq):
    """Flips the codon characters from one seq to another (PRIVATE)."""
    a, b = ('', '')
    for char1, char2 in zip(codon_seq, target_seq):
        if char1 == ' ':
            a += char1
            b += char2
        else:
            a += char2
            b += char1
    return (a, b)