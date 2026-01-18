import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlastTabParser:
    """Parser for the BLAST tabular format."""

    def __init__(self, handle, comments=False, fields=_DEFAULT_FIELDS):
        """Initialize the class."""
        self.handle = handle
        self.has_comments = comments
        self.fields = self._prep_fields(fields)
        self.line = self.handle.readline().strip()

    def __iter__(self):
        """Iterate over BlastTabParser, yields query results."""
        if not self.line:
            return
        elif self.has_comments:
            iterfunc = self._parse_commented_qresult
        else:
            if self.line.startswith('#'):
                raise ValueError("Encountered unexpected character '#' at the beginning of a line. Set comments=True if the file is a commented file.")
            iterfunc = self._parse_qresult
        yield from iterfunc()

    def _prep_fields(self, fields):
        """Validate and format the given fields for use by the parser (PRIVATE)."""
        if isinstance(fields, str):
            fields = fields.strip().split(' ')
        if 'std' in fields:
            idx = fields.index('std')
            fields = fields[:idx] + _DEFAULT_FIELDS + fields[idx + 1:]
        if not set(fields).intersection(_MIN_QUERY_FIELDS) or not set(fields).intersection(_MIN_HIT_FIELDS):
            raise ValueError('Required query and/or hit ID field not found.')
        return fields

    def _parse_commented_qresult(self):
        """Yield ``QueryResult`` objects from a commented file (PRIVATE)."""
        while True:
            comments = self._parse_comments()
            if comments:
                try:
                    self.fields = comments['fields']
                    qres_iter = self._parse_qresult()
                except KeyError:
                    assert 'fields' not in comments
                    qres_iter = iter([QueryResult()])
                for qresult in qres_iter:
                    for key, value in comments.items():
                        setattr(qresult, key, value)
                    yield qresult
            else:
                break

    def _parse_comments(self):
        """Return a dictionary containing tab file comments (PRIVATE)."""
        comments = {}
        while True:
            if 'BLAST' in self.line and 'processed' not in self.line:
                program_line = self.line[len(' #'):].split(' ')
                comments['program'] = program_line[0].lower()
                comments['version'] = program_line[1]
            elif 'Query' in self.line:
                query_line = self.line[len('# Query: '):].split(' ', 1)
                comments['id'] = query_line[0]
                if len(query_line) == 2:
                    comments['description'] = query_line[1]
            elif 'Database' in self.line:
                comments['target'] = self.line[len('# Database: '):]
            elif 'RID' in self.line:
                comments['rid'] = self.line[len('# RID: '):]
            elif 'Fields' in self.line:
                comments['fields'] = self._parse_fields_line()
            elif ' hits found' in self.line or 'processed' in self.line:
                self.line = self.handle.readline().strip()
                return comments
            self.line = self.handle.readline()
            if not self.line:
                return comments
            else:
                self.line = self.line.strip()

    def _parse_fields_line(self):
        """Return column short names line from 'Fields' comment line (PRIVATE)."""
        raw_field_str = self.line[len('# Fields: '):]
        long_fields = raw_field_str.split(', ')
        fields = [_LONG_SHORT_MAP[long_name] for long_name in long_fields]
        return self._prep_fields(fields)

    def _parse_result_row(self):
        """Return a dictionary of parsed row values (PRIVATE)."""
        fields = self.fields
        columns = self.line.strip().split('\t')
        if len(fields) != len(columns):
            raise ValueError('Expected %i columns, found: %i' % (len(fields), len(columns)))
        qresult, hit, hsp, frag = ({}, {}, {}, {})
        for idx, value in enumerate(columns):
            sname = fields[idx]
            in_mapping = False
            for parsed_dict, mapping in ((qresult, _COLUMN_QRESULT), (hit, _COLUMN_HIT), (hsp, _COLUMN_HSP), (frag, _COLUMN_FRAG)):
                if sname in mapping:
                    attr_name, caster = mapping[sname]
                    if caster is not str:
                        value = caster(value)
                    parsed_dict[attr_name] = value
                    in_mapping = True
            if not in_mapping:
                assert sname not in _SUPPORTED_FIELDS
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp, 'frag': frag}

    def _get_id(self, parsed):
        """Return the value used for a QueryResult or Hit ID from a parsed row (PRIVATE)."""
        id_cache = parsed.get('id')
        if id_cache is None and 'id_all' in parsed:
            id_cache = parsed.get('id_all')[0]
        if id_cache is None:
            id_cache = parsed.get('accession')
        if id_cache is None:
            id_cache = parsed.get('accession_version')
        return id_cache

    def _parse_qresult(self):
        """Yield QueryResult objects (PRIVATE)."""
        state_EOF = 0
        state_QRES_NEW = 1
        state_QRES_SAME = 3
        state_HIT_NEW = 2
        state_HIT_SAME = 4
        qres_state = None
        hit_state = None
        file_state = None
        cur_qid = None
        cur_hid = None
        prev_qid = None
        prev_hid = None
        cur, prev = (None, None)
        hit_list, hsp_list = ([], [])
        while True:
            if cur is not None:
                prev = cur
                prev_qid = cur_qid
                prev_hid = cur_hid
            if self.line and (not self.line.startswith('#')):
                cur = self._parse_result_row()
                cur_qid = self._get_id(cur['qresult'])
                cur_hid = self._get_id(cur['hit'])
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
                    for seq_type in ('query', 'hit'):
                        if attr == seq_type + '_start':
                            value = min(value, prev['frag'][seq_type + '_end']) - 1
                        elif attr == seq_type + '_end':
                            value = max(value, prev['frag'][seq_type + '_start'])
                    setattr(frag, attr, value)
                for seq_type in ('hit', 'query'):
                    frame = self._get_frag_frame(frag, seq_type, prev['frag'])
                    setattr(frag, '%s_frame' % seq_type, frame)
                    strand = self._get_frag_strand(frag, seq_type, prev['frag'])
                    setattr(frag, '%s_strand' % seq_type, strand)
                hsp = HSP([frag])
                for attr, value in prev['hsp'].items():
                    setattr(hsp, attr, value)
                hsp_list.append(hsp)
                if hit_state == state_HIT_NEW:
                    hit = Hit(hsp_list)
                    for attr, value in prev['hit'].items():
                        if attr != 'id_all':
                            setattr(hit, attr, value)
                        else:
                            setattr(hit, '_id_alt', value[1:])
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
            self.line = self.handle.readline().strip()

    def _get_frag_frame(self, frag, seq_type, parsedict):
        """Return fragment frame for given object (PRIVATE).

        Returns ``HSPFragment`` frame given the object, its sequence type,
        and its parsed dictionary values.
        """
        assert seq_type in ('query', 'hit')
        frame = getattr(frag, '%s_frame' % seq_type, None)
        if frame is not None:
            return frame
        elif 'frames' in parsedict:
            idx = 0 if seq_type == 'query' else 1
            return int(parsedict['frames'].split('/')[idx])

    def _get_frag_strand(self, frag, seq_type, parsedict):
        """Return fragment strand for given object (PRIVATE).

        Returns ``HSPFragment`` strand given the object, its sequence type,
        and its parsed dictionary values.
        """
        assert seq_type in ('query', 'hit')
        strand = getattr(frag, '%s_strand' % seq_type, None)
        if strand is not None:
            return strand
        else:
            start = parsedict.get('%s_start' % seq_type)
            end = parsedict.get('%s_end' % seq_type)
            if start is not None and end is not None:
                return 1 if start <= end else -1