import re
from ._base import _BaseExonerateParser, _STRAND_MAP
from .exonerate_vulgar import ExonerateVulgarIndexer
class ExonerateCigarParser(_BaseExonerateParser):
    """Parser for Exonerate cigar strings."""
    _ALN_MARK = 'cigar'

    def parse_alignment_block(self, header):
        """Parse alignment block for cigar format, return query results, hits, hsps."""
        qresult = header['qresult']
        hit = header['hit']
        hsp = header['hsp']
        self.read_until(lambda line: line.startswith('cigar'))
        cigars = re.search(_RE_CIGAR, self.line)
        if self.has_c4_alignment:
            assert qresult['id'] == cigars.group(1)
            assert hsp['query_start'] == cigars.group(2)
            assert hsp['query_end'] == cigars.group(3)
            assert hsp['query_strand'] == cigars.group(4)
            assert hit['id'] == cigars.group(5)
            assert hsp['hit_start'] == cigars.group(6)
            assert hsp['hit_end'] == cigars.group(7)
            assert hsp['hit_strand'] == cigars.group(8)
            assert hsp['score'] == cigars.group(9)
        else:
            qresult['id'] = cigars.group(1)
            hsp['query_start'] = cigars.group(2)
            hsp['query_end'] = cigars.group(3)
            hsp['query_strand'] = cigars.group(4)
            hit['id'] = cigars.group(5)
            hsp['hit_start'] = cigars.group(6)
            hsp['hit_end'] = cigars.group(7)
            hsp['hit_strand'] = cigars.group(8)
            hsp['score'] = cigars.group(9)
        hsp['query_strand'] = _STRAND_MAP[hsp['query_strand']]
        hsp['hit_strand'] = _STRAND_MAP[hsp['hit_strand']]
        qstart = int(hsp['query_start'])
        qend = int(hsp['query_end'])
        hstart = int(hsp['hit_start'])
        hend = int(hsp['hit_end'])
        hsp['query_start'] = min(qstart, qend)
        hsp['query_end'] = max(qstart, qend)
        hsp['hit_start'] = min(hstart, hend)
        hsp['hit_end'] = max(hstart, hend)
        hsp['score'] = int(hsp['score'])
        hsp['cigar_comp'] = cigars.group(10)
        hsp['query_ranges'] = [(hsp['query_start'], hsp['query_end'])]
        hsp['hit_ranges'] = [(hsp['hit_start'], hsp['hit_end'])]
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp}