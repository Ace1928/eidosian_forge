from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
def _parse_row(self):
    """Return a dictionary of parsed row values (PRIVATE)."""
    assert self.line
    cols = [x for x in self.line.strip().split(' ') if x]
    if len(cols) > 23:
        cols[22] = ' '.join(cols[22:])
    elif len(cols) < 23:
        cols.append('')
        assert len(cols) == 23
    qresult = {}
    qresult['id'] = cols[3]
    qresult['accession'] = cols[4]
    qresult['seq_len'] = int(cols[5])
    hit = {}
    hit['id'] = cols[0]
    hit['accession'] = cols[1]
    hit['seq_len'] = int(cols[2])
    hit['evalue'] = float(cols[6])
    hit['bitscore'] = float(cols[7])
    hit['bias'] = float(cols[8])
    hit['description'] = cols[22]
    hsp = {}
    hsp['domain_index'] = int(cols[9])
    hsp['evalue_cond'] = float(cols[11])
    hsp['evalue'] = float(cols[12])
    hsp['bitscore'] = float(cols[13])
    hsp['bias'] = float(cols[14])
    hsp['env_start'] = int(cols[19]) - 1
    hsp['env_end'] = int(cols[20])
    hsp['acc_avg'] = float(cols[21])
    frag = {}
    frag['hit_strand'] = frag['query_strand'] = 0
    frag['hit_start'] = int(cols[15]) - 1
    frag['hit_end'] = int(cols[16])
    frag['query_start'] = int(cols[17]) - 1
    frag['query_end'] = int(cols[18])
    frag['molecule_type'] = 'protein'
    if not self.hmm_as_hit:
        frag['hit_end'], frag['query_end'] = (frag['query_end'], frag['hit_end'])
        frag['hit_start'], frag['query_start'] = (frag['query_start'], frag['hit_start'])
    return {'qresult': qresult, 'hit': hit, 'hsp': hsp, 'frag': frag}