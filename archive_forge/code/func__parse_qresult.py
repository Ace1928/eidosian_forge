import re
from xml.etree import ElementTree
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_qresult(self):
    """Parse query results (PRIVATE)."""
    for event, elem in self.xml_iter:
        if event == 'end' and elem.tag == self.NS + 'protein':
            seq = elem.find(self.NS + 'sequence')
            query_seq = seq.text
            xref = elem.find(self.NS + 'xref')
            query_id = xref.attrib['id']
            query_desc = xref.attrib['name']
            hit_list = []
            for hit_new in self._parse_hit(elem.find(self.NS + 'matches'), query_id, query_seq):
                for hit in hit_list:
                    if hit.id == hit_new.id:
                        for hsp in hit_new.hsps:
                            hit.hsps.append(hsp)
                        break
                else:
                    hit_list.append(hit_new)
            qresult = QueryResult(hit_list, query_id)
            setattr(qresult, 'description', query_desc)
            for key, value in self._meta.items():
                setattr(qresult, key, value)
            yield qresult