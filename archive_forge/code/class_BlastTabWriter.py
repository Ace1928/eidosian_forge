import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class BlastTabWriter:
    """Writer for blast-tab output format."""

    def __init__(self, handle, comments=False, fields=_DEFAULT_FIELDS):
        """Initialize the class."""
        self.handle = handle
        self.has_comments = comments
        self.fields = fields

    def write_file(self, qresults):
        """Write to the handle, return how many QueryResult objects were written."""
        handle = self.handle
        qresult_counter, hit_counter, hsp_counter, frag_counter = (0, 0, 0, 0)
        for qresult in qresults:
            if self.has_comments:
                handle.write(self._build_comments(qresult))
            if qresult:
                handle.write(self._build_rows(qresult))
                if not self.has_comments:
                    qresult_counter += 1
                hit_counter += len(qresult)
                hsp_counter += sum((len(hit) for hit in qresult))
                frag_counter += sum((len(hit.fragments) for hit in qresult))
            if self.has_comments:
                qresult_counter += 1
        if self.has_comments:
            handle.write('# BLAST processed %i queries' % qresult_counter)
        return (qresult_counter, hit_counter, hsp_counter, frag_counter)

    def _build_rows(self, qresult):
        """Return a string containing tabular rows of the QueryResult object (PRIVATE)."""
        coordinates = {'qstart', 'qend', 'sstart', 'send'}
        qresult_lines = ''
        for hit in qresult:
            for hsp in hit:
                line = []
                for field in self.fields:
                    if field in _COLUMN_QRESULT:
                        value = getattr(qresult, _COLUMN_QRESULT[field][0])
                    elif field in _COLUMN_HIT:
                        if field == 'sallseqid':
                            value = getattr(hit, 'id_all')
                        else:
                            value = getattr(hit, _COLUMN_HIT[field][0])
                    elif field == 'frames':
                        value = '%i/%i' % (hsp.query_frame, hsp.hit_frame)
                    elif field in _COLUMN_HSP:
                        try:
                            value = getattr(hsp, _COLUMN_HSP[field][0])
                        except AttributeError:
                            attr = _COLUMN_HSP[field][0]
                            _augment_blast_hsp(hsp, attr)
                            value = getattr(hsp, attr)
                    elif field in _COLUMN_FRAG:
                        value = getattr(hsp, _COLUMN_FRAG[field][0])
                    else:
                        assert field not in _SUPPORTED_FIELDS
                        continue
                    if field in coordinates:
                        value = self._adjust_coords(field, value, hsp)
                    value = self._adjust_output(field, value)
                    line.append(value)
                hsp_line = '\t'.join(line)
                qresult_lines += hsp_line + '\n'
        return qresult_lines

    def _adjust_coords(self, field, value, hsp):
        """Adjust start and end coordinates according to strand (PRIVATE)."""
        assert field in ('qstart', 'qend', 'sstart', 'send')
        seq_type = 'query' if field.startswith('q') else 'hit'
        strand = getattr(hsp, '%s_strand' % seq_type, None)
        if strand is None:
            raise ValueError('Required attribute %r not found.' % ('%s_strand' % seq_type))
        if strand < 0:
            if field.endswith('start'):
                value = getattr(hsp, '%s_end' % seq_type)
            elif field.endswith('end'):
                value = getattr(hsp, '%s_start' % seq_type) + 1
        elif field.endswith('start'):
            value += 1
        return value

    def _adjust_output(self, field, value):
        """Adjust formatting of given field and value to mimic native tab output (PRIVATE)."""
        if field in ('qseq', 'sseq'):
            value = str(value.seq)
        elif field == 'evalue':
            if value < 1e-180:
                value = '0.0'
            elif value < 1e-99:
                value = '%2.0e' % value
            elif value < 0.0009:
                value = '%3.0e' % value
            elif value < 0.1:
                value = '%4.3f' % value
            elif value < 1.0:
                value = '%3.2f' % value
            elif value < 10.0:
                value = '%2.1f' % value
            else:
                value = '%5.0f' % value
        elif field in ('pident', 'ppos'):
            value = '%.2f' % value
        elif field == 'bitscore':
            if value > 9999:
                value = '%4.3e' % value
            elif value > 99.9:
                value = '%4.0d' % value
            else:
                value = '%4.1f' % value
        elif field in ('qcovhsp', 'qcovs'):
            value = '%.0f' % value
        elif field == 'salltitles':
            value = '<>'.join(value)
        elif field in ('sallseqid', 'sallacc', 'staxids', 'sscinames', 'scomnames', 'sblastnames', 'sskingdoms'):
            value = ';'.join(value)
        else:
            value = str(value)
        return value

    def _build_comments(self, qres):
        """Return QueryResult tabular comment as a string (PRIVATE)."""
        comments = []
        inv_field_map = {v: k for k, v in _LONG_SHORT_MAP.items()}
        program = qres.program.upper()
        try:
            version = qres.version
        except AttributeError:
            program_line = '# %s' % program
        else:
            program_line = f'# {program} {version}'
        comments.append(program_line)
        if qres.description is None:
            comments.append('# Query: %s' % qres.id)
        else:
            comments.append(f'# Query: {qres.id} {qres.description}')
        try:
            comments.append('# RID: %s' % qres.rid)
        except AttributeError:
            pass
        comments.append('# Database: %s' % qres.target)
        if qres:
            comments.append('# Fields: %s' % ', '.join((inv_field_map[field] for field in self.fields)))
        comments.append('# %i hits found' % len(qres))
        return '\n'.join(comments) + '\n'