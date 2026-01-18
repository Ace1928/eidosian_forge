import re
from Bio.SearchIO._utils import read_forward, removesuffix
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def _create_hits(self, hit_attrs, qid, qdesc):
    """Parse a HMMER3 hsp block, beginning with the hsp table (PRIVATE)."""
    self._read_until(lambda line: line.startswith(('Internal pipeline', '>>')))
    hit_list = []
    while True:
        if self.line.startswith('Internal pipeline'):
            assert len(hit_attrs) == 0
            return hit_list
        assert self.line.startswith('>>')
        hid, hdesc = self.line[len('>> '):].split('  ', 1)
        hdesc = hdesc.strip()
        self._read_until(lambda line: line.startswith((' ---   ------ ----- --------', '   [No individual domains')))
        self.line = read_forward(self.handle)
        hsp_list = []
        while True:
            if self.line.startswith('   [No targets detected that satisfy') or self.line.startswith('   [No individual domains') or self.line.startswith('Internal pipeline statistics summary:') or self.line.startswith('  Alignments for each domain:') or self.line.startswith('>>'):
                hit_attr = hit_attrs.pop(0)
                hit = Hit(hsp_list)
                for attr, value in hit_attr.items():
                    if attr == 'description':
                        cur_val = getattr(hit, attr)
                        if cur_val and value and cur_val.startswith(value):
                            continue
                    setattr(hit, attr, value)
                if not hit:
                    hit.query_description = qdesc
                hit_list.append(hit)
                break
            parsed = [x for x in self.line.strip().split(' ') if x]
            assert len(parsed) == 16
            frag = HSPFragment(hid, qid)
            if qdesc:
                frag.query_description = qdesc
            if hdesc:
                frag.hit_description = hdesc
            frag.molecule_type = 'protein'
            if self._meta.get('program') == 'hmmscan':
                frag.hit_start = int(parsed[6]) - 1
                frag.hit_end = int(parsed[7])
                frag.query_start = int(parsed[9]) - 1
                frag.query_end = int(parsed[10])
            elif self._meta.get('program') in ['hmmsearch', 'phmmer']:
                frag.hit_start = int(parsed[9]) - 1
                frag.hit_end = int(parsed[10])
                frag.query_start = int(parsed[6]) - 1
                frag.query_end = int(parsed[7])
            frag.hit_strand = frag.query_strand = 0
            hsp = HSP([frag])
            hsp.domain_index = int(parsed[0])
            hsp.is_included = parsed[1] == '!'
            hsp.bitscore = float(parsed[2])
            hsp.bias = float(parsed[3])
            hsp.evalue_cond = float(parsed[4])
            hsp.evalue = float(parsed[5])
            if self._meta.get('program') == 'hmmscan':
                hsp.hit_endtype = parsed[8]
                hsp.query_endtype = parsed[11]
            elif self._meta.get('program') in ['hmmsearch', 'phmmer']:
                hsp.hit_endtype = parsed[11]
                hsp.query_endtype = parsed[8]
            hsp.env_start = int(parsed[12]) - 1
            hsp.env_end = int(parsed[13])
            hsp.env_endtype = parsed[14]
            hsp.acc_avg = float(parsed[15])
            hsp_list.append(hsp)
            self.line = read_forward(self.handle)
        if self.line.startswith('  Alignments for each domain:'):
            self._parse_aln_block(hid, hit.hsps)