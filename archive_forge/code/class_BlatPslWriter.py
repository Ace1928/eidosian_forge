import re
from math import log
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlatPslWriter:
    """Writer for the blat-psl format."""

    def __init__(self, handle, header=False, pslx=False):
        """Initialize the class."""
        self.handle = handle
        self.header = header
        self.pslx = pslx

    def write_file(self, qresults):
        """Write query results to file."""
        handle = self.handle
        qresult_counter, hit_counter, hsp_counter, frag_counter = (0, 0, 0, 0)
        if self.header:
            handle.write(self._build_header())
        for qresult in qresults:
            if qresult:
                handle.write(self._build_row(qresult))
                qresult_counter += 1
                hit_counter += len(qresult)
                hsp_counter += sum((len(hit) for hit in qresult))
                frag_counter += sum((len(hit.fragments) for hit in qresult))
        return (qresult_counter, hit_counter, hsp_counter, frag_counter)

    def _build_header(self):
        """Build header, tab-separated string (PRIVATE)."""
        header = 'psLayout version 3\n'
        header += "\nmatch\tmis- \trep. \tN's\tQ gap\tQ gap\tT gap\tT gap\tstrand\tQ        \tQ   \tQ    \tQ  \tT        \tT   \tT    \tT  \tblock\tblockSizes \tqStarts\t tStarts\n     \tmatch\tmatch\t   \tcount\tbases\tcount\tbases\t      \tname     \tsize\tstart\tend\tname     \tsize\tstart\tend\tcount\n%s\n" % ('-' * 159)
        return header

    def _build_row(self, qresult):
        """Return a string or one row or more of the QueryResult object (PRIVATE)."""
        qresult_lines = []
        for hit in qresult:
            for hsp in hit.hsps:
                query_is_protein = getattr(hsp, 'query_is_protein', False)
                blocksize_multiplier = 3 if query_is_protein else 1
                line = []
                line.append(hsp.match_num)
                line.append(hsp.mismatch_num)
                line.append(hsp.match_rep_num)
                line.append(hsp.n_num)
                line.append(hsp.query_gapopen_num)
                line.append(hsp.query_gap_num)
                line.append(hsp.hit_gapopen_num)
                line.append(hsp.hit_gap_num)
                eff_query_spans = [blocksize_multiplier * s for s in hsp.query_span_all]
                if hsp.hit_span_all != eff_query_spans:
                    raise ValueError('HSP hit span and query span values do not match.')
                block_sizes = hsp.query_span_all
                if hsp[0].query_strand >= 0:
                    strand = '+'
                else:
                    strand = '-'
                qstarts = _reorient_starts([x[0] for x in hsp.query_range_all], hsp.query_span_all, qresult.seq_len, hsp[0].query_strand)
                if hsp[0].hit_strand == 1:
                    hstrand = 1
                    if hsp._has_hit_strand:
                        strand += '+'
                else:
                    hstrand = -1
                    strand += '-'
                hstarts = _reorient_starts([x[0] for x in hsp.hit_range_all], hsp.hit_span_all, hit.seq_len, hstrand)
                line.append(strand)
                line.append(qresult.id)
                line.append(qresult.seq_len)
                line.append(hsp.query_start)
                line.append(hsp.query_end)
                line.append(hit.id)
                line.append(hit.seq_len)
                line.append(hsp.hit_start)
                line.append(hsp.hit_end)
                line.append(len(hsp))
                line.append(','.join((str(x) for x in block_sizes)) + ',')
                line.append(','.join((str(x) for x in qstarts)) + ',')
                line.append(','.join((str(x) for x in hstarts)) + ',')
                if self.pslx:
                    line.append(','.join((str(x.seq) for x in hsp.query_all)) + ',')
                    line.append(','.join((str(x.seq) for x in hsp.hit_all)) + ',')
                qresult_lines.append('\t'.join((str(x) for x in line)))
        return '\n'.join(qresult_lines) + '\n'