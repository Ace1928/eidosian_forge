import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
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