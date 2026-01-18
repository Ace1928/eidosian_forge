import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _comp_split_codons(hsp, seq_type, scodon_moves):
    """Compute positions of split codons, store in given HSP dictionary (PRIVATE)."""
    scodons = []
    for idx in range(len(scodon_moves[seq_type])):
        pair = scodon_moves[seq_type][idx]
        if not any(pair):
            continue
        else:
            assert not all(pair)
        a, b = pair
        anchor_pair = hsp['%s_ranges' % seq_type][idx // 2]
        strand = 1 if hsp['%s_strand' % seq_type] >= 0 else -1
        if a:
            func = max if strand == 1 else min
            anchor = func(anchor_pair)
            start_c, end_c = (anchor + a * strand * -1, anchor)
        elif b:
            func = min if strand == 1 else max
            anchor = func(anchor_pair)
            start_c, end_c = (anchor + b * strand, anchor)
        scodons.append((min(start_c, end_c), max(start_c, end_c)))
    return scodons