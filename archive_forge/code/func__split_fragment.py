import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _split_fragment(frag):
    """Split one HSPFragment containing frame-shifted alignment into two (PRIVATE)."""
    simil = frag.aln_annotation['similarity']
    assert simil.count('#') > 0
    split_frags = []
    qstep = 1 if frag.query_strand >= 0 else -1
    hstep = 1 if frag.hit_strand >= 0 else -1
    qpos = min(frag.query_range) if qstep >= 0 else max(frag.query_range)
    hpos = min(frag.hit_range) if hstep >= 0 else max(frag.hit_range)
    abs_pos = 0
    while simil:
        try:
            shifts = re.search(_RE_SHIFTS, simil).group(1)
            s_start = simil.find(shifts)
            s_stop = s_start + len(shifts)
            split = frag[abs_pos:abs_pos + s_start]
        except AttributeError:
            shifts = ''
            s_start = 0
            s_stop = len(simil)
            split = frag[abs_pos:]
        qstart, hstart = (qpos, hpos)
        qpos += (len(split) - sum((split.query.seq.count(x) for x in ('-', '<', '>')))) * qstep
        hpos += (len(split) - sum((split.hit.seq.count(x) for x in ('-', '<', '>')))) * hstep
        split.hit_start = min(hstart, hpos)
        split.query_start = min(qstart, qpos)
        split.hit_end = max(hstart, hpos)
        split.query_end = max(qstart, qpos)
        abs_slice = slice(abs_pos + s_start, abs_pos + s_stop)
        if len(frag.aln_annotation) == 2:
            seqs = (frag[abs_slice].query.seq, frag[abs_slice].hit.seq)
        elif len(frag.aln_annotation) == 3:
            seqs = (frag[abs_slice].aln_annotation['query_annotation'], frag[abs_slice].aln_annotation['hit_annotation'])
        if '#' in seqs[0]:
            qpos += len(shifts) * qstep
        elif '#' in seqs[1]:
            hpos += len(shifts) * hstep
        _set_frame(split)
        split_frags.append(split)
        simil = simil[s_stop:]
        abs_pos += s_stop
    return split_frags