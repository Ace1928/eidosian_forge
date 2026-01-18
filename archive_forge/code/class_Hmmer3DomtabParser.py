from itertools import chain
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from .hmmer3_tab import Hmmer3TabParser, Hmmer3TabIndexer
class Hmmer3DomtabParser(Hmmer3TabParser):
    """Base hmmer3-domtab iterator."""

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

    def _parse_qresult(self):
        """Return QueryResult objects (PRIVATE)."""
        state_EOF = 0
        state_QRES_NEW = 1
        state_QRES_SAME = 3
        state_HIT_NEW = 2
        state_HIT_SAME = 4
        qres_state = None
        hit_state = None
        file_state = None
        prev_qid = None
        prev_hid = None
        cur, prev = (None, None)
        hit_list, hsp_list = ([], [])
        cur_qid = None
        cur_hid = None
        while True:
            if cur is not None:
                prev = cur
                prev_qid = cur_qid
                prev_hid = cur_hid
            if self.line and (not self.line.startswith('#')):
                cur = self._parse_row()
                cur_qid = cur['qresult']['id']
                cur_hid = cur['hit']['id']
            else:
                file_state = state_EOF
                cur_qid, cur_hid = (None, None)
            if prev_qid != cur_qid:
                qres_state = state_QRES_NEW
            else:
                qres_state = state_QRES_SAME
            if prev_hid != cur_hid or qres_state == state_QRES_NEW:
                hit_state = state_HIT_NEW
            else:
                hit_state = state_HIT_SAME
            if prev is not None:
                frag = HSPFragment(prev_hid, prev_qid)
                for attr, value in prev['frag'].items():
                    setattr(frag, attr, value)
                hsp = HSP([frag])
                for attr, value in prev['hsp'].items():
                    setattr(hsp, attr, value)
                hsp_list.append(hsp)
                if hit_state == state_HIT_NEW:
                    hit = Hit(hsp_list)
                    for attr, value in prev['hit'].items():
                        setattr(hit, attr, value)
                    hit_list.append(hit)
                    hsp_list = []
                if qres_state == state_QRES_NEW or file_state == state_EOF:
                    qresult = QueryResult(hit_list, prev_qid)
                    for attr, value in prev['qresult'].items():
                        setattr(qresult, attr, value)
                    yield qresult
                    if file_state == state_EOF:
                        break
                    hit_list = []
            self.line = self.handle.readline()