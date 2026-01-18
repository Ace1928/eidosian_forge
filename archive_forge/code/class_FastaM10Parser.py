import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class FastaM10Parser:
    """Parser for Bill Pearson's FASTA suite's -m 10 output."""

    def __init__(self, handle, __parse_hit_table=False):
        """Initialize the class."""
        self.handle = handle
        self._preamble = self._parse_preamble()

    def __iter__(self):
        """Iterate over FastaM10Parser object yields query results."""
        for qresult in self._parse_qresult():
            qresult.description = qresult.description
            yield qresult

    def _parse_preamble(self):
        """Parse the Fasta preamble for Fasta flavor and version (PRIVATE)."""
        preamble = {}
        while True:
            line = self.handle.readline()
            if line.startswith('Query'):
                break
            elif line.startswith(' version'):
                preamble['version'] = line.split(' ')[2]
            else:
                flav_match = re.match(_RE_FLAVS, line.lower())
                if flav_match:
                    preamble['program'] = flav_match.group(0)
        self.line = line
        return preamble

    def __parse_hit_table(self):
        """Parse hit table rows."""
        hit_rows = []
        while True:
            line = self.handle.readline()
            if not line or line.strip():
                break
            hit_rows.append('')
        self.line = line
        return hit_rows

    def _parse_qresult(self):
        """Parse query result (PRIVATE)."""
        qresult = None
        hit_rows = []
        state_QRES_NEW = 1
        state_QRES_HITTAB = 3
        state_QRES_CONTENT = 5
        state_QRES_END = 7
        line = self.line
        while True:
            if line.startswith('The best scores are:'):
                qres_state = state_QRES_HITTAB
            elif line.strip() == '>>>///' or not line:
                qres_state = state_QRES_END
            elif not line.startswith('>>>') and '>>>' in line:
                qres_state = state_QRES_NEW
            elif line.startswith('>>>') and line.strip() != '>>><<<':
                qres_state = state_QRES_CONTENT
            else:
                qres_state = None
            if qres_state is not None:
                if qres_state == state_QRES_HITTAB:
                    hit_rows = self.__parse_hit_table()
                    line = self.handle.readline()
                elif qres_state == state_QRES_END:
                    yield _set_qresult_hits(qresult, hit_rows)
                    break
                elif qres_state == state_QRES_NEW:
                    if qresult is not None:
                        yield _set_qresult_hits(qresult, hit_rows)
                    regx = re.search(_RE_ID_DESC_SEQLEN, line)
                    query_id = regx.group(1)
                    seq_len = regx.group(3)
                    desc = regx.group(2)
                    qresult = QueryResult(id=query_id)
                    qresult.seq_len = int(seq_len)
                    line = self.handle.readline()
                    qresult.target = [x for x in line.split(' ') if x][1].strip()
                    if desc is not None:
                        qresult.description = desc
                    for key, value in self._preamble.items():
                        setattr(qresult, key, value)
                    line = self.handle.readline()
                elif qres_state == state_QRES_CONTENT:
                    assert line[3:].startswith(qresult.id), line
                    for hit, strand in self._parse_hit(query_id):
                        hit.description = hit.description
                        hit.query_description = qresult.description
                        if hit.id not in qresult:
                            qresult.append(hit)
                        else:
                            for hsp in hit.hsps:
                                assert strand != hsp.query_strand
                                qresult[hit.id].append(hsp)
                    line = self.line
            else:
                line = self.handle.readline()
        self.line = line

    def _parse_hit(self, query_id):
        """Parse hit on query identifier (PRIVATE)."""
        while True:
            line = self.handle.readline()
            if line.startswith('>>'):
                break
        state = _STATE_NONE
        strand = None
        hsp_list = []
        hsp = None
        parsed_hsp = None
        hit_desc = None
        seq_len = None
        while True:
            self.line = self.handle.readline()
            if self.line.strip() in ['>>><<<', '>>>///'] or (not self.line.startswith('>>>') and '>>>' in self.line):
                if state == _STATE_HIT_BLOCK:
                    parsed_hsp['hit']['seq'] += line.strip()
                elif state == _STATE_CONS_BLOCK:
                    hsp.aln_annotation['similarity'] += line.strip('\r\n')
                _set_hsp_seqs(hsp, parsed_hsp, self._preamble['program'])
                hit = Hit(hsp_list)
                hit.description = hit_desc
                hit.seq_len = seq_len
                yield (hit, strand)
                hsp_list = []
                break
            elif line.startswith('>>'):
                if hsp_list:
                    _set_hsp_seqs(hsp, parsed_hsp, self._preamble['program'])
                    hit = Hit(hsp_list)
                    hit.description = hit_desc
                    hit.seq_len = seq_len
                    yield (hit, strand)
                    hsp_list = []
                try:
                    hit_id, hit_desc = line[2:].strip().split(' ', 1)
                except ValueError:
                    hit_id = line[2:].strip().split(' ', 1)[0]
                    hit_desc = ''
                frag = HSPFragment(hit_id, query_id)
                hsp = HSP([frag])
                hsp_list.append(hsp)
                state = _STATE_NONE
                parsed_hsp = {'query': {}, 'hit': {}}
            elif line.startswith('>--'):
                _set_hsp_seqs(hsp, parsed_hsp, self._preamble['program'])
                frag = HSPFragment(hit_id, query_id)
                hsp = HSP([frag])
                hsp_list.append(hsp)
                state = _STATE_NONE
                parsed_hsp = {'query': {}, 'hit': {}}
            elif line.startswith('>'):
                if state == _STATE_NONE:
                    if not query_id.startswith(line[1:].split(' ')[0]):
                        raise ValueError(f'{query_id!r} vs {line!r}')
                    state = _STATE_QUERY_BLOCK
                    parsed_hsp['query']['seq'] = ''
                elif state == _STATE_QUERY_BLOCK:
                    assert hit_id.startswith(line[1:].split(' ')[0])
                    state = _STATE_HIT_BLOCK
                    parsed_hsp['hit']['seq'] = ''
            elif line.startswith('; al_cons'):
                state = _STATE_CONS_BLOCK
                hsp.fragment.aln_annotation['similarity'] = ''
            elif line.startswith(';'):
                regx = re.search(_RE_ATTR, line.strip())
                name = regx.group(1)
                value = regx.group(2)
                if state == _STATE_NONE:
                    if name in _HSP_ATTR_MAP:
                        attr_name, caster = _HSP_ATTR_MAP[name]
                        if caster is not str:
                            value = caster(value)
                        if name in ['_ident', '_sim']:
                            value *= 100
                        setattr(hsp, attr_name, value)
                elif state == _STATE_QUERY_BLOCK:
                    parsed_hsp['query'][name] = value
                elif state == _STATE_HIT_BLOCK:
                    if name == '_len':
                        seq_len = int(value)
                    else:
                        parsed_hsp['hit'][name] = value
                else:
                    raise ValueError('Unexpected line: %r' % line)
            else:
                assert '>' not in line
                if state == _STATE_HIT_BLOCK:
                    parsed_hsp['hit']['seq'] += line.strip()
                elif state == _STATE_QUERY_BLOCK:
                    parsed_hsp['query']['seq'] += line.strip()
                elif state == _STATE_CONS_BLOCK:
                    hsp.fragment.aln_annotation['similarity'] += line.strip('\r\n')
                else:
                    raise ValueError('Unexpected line: %r' % line)
            line = self.line