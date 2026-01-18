import re
from functools import reduce
from abc import ABC, abstractmethod
from typing import Optional, Type
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from Bio.SeqUtils import seq1
def _adjust_aa_seq(fraglist):
    """Transform 3-letter AA codes of input fragments to one-letter codes (PRIVATE).

    Argument fraglist should be a list of HSPFragments objects.
    """
    custom_map = {'***': '*', '<->': '-'}
    hsp_hstart = fraglist[0].hit_start
    hsp_qstart = fraglist[0].query_start
    frag_phases = _get_fragments_phase(fraglist)
    for frag, phase in zip(fraglist, frag_phases):
        assert frag.query_strand == 0 or frag.hit_strand == 0
        hstep = 1 if frag.hit_strand >= 0 else -1
        frag.phase = phase
        qseq = str(frag.query.seq)
        q_triplets_pre, q_triplets, q_triplets_post = _make_triplets(qseq, phase)
        hseq = str(frag.hit.seq)
        h_triplets_pre, h_triplets, h_triplets_post = _make_triplets(hseq, phase)
        hseq1_pre = 'X' if h_triplets_pre else ''
        hseq1_post = 'X' if h_triplets_post else ''
        hseq1 = seq1(''.join(h_triplets), custom_map=custom_map)
        hstart = hsp_hstart + len(hseq1_pre) * hstep
        hend = hstart + len(hseq1.replace('-', '')) * hstep
        qseq1_pre = 'X' if q_triplets_pre else ''
        qseq1_post = 'X' if q_triplets_post else ''
        qseq1 = seq1(''.join(q_triplets), custom_map=custom_map)
        qstart = hsp_qstart + len(qseq1_pre)
        qend = qstart + len(qseq1.replace('-', ''))
        frag.hit = None
        frag.query = None
        frag.hit = hseq1_pre + hseq1 + hseq1_post
        frag.query = qseq1_pre + qseq1 + qseq1_post
        if frag.query_strand == 0:
            frag.query_start, frag.query_end = (qstart, qend)
        elif frag.hit_strand == 0:
            frag.hit_start, frag.hit_end = (hstart, hend)
        for annot, annotseq in frag.aln_annotation.items():
            pre, intact, post = _make_triplets(annotseq, phase)
            frag.aln_annotation[annot] = list(filter(None, [pre])) + intact + list(filter(None, [post]))
        hsp_hstart, hsp_qstart = (hend, qend)
    return fraglist