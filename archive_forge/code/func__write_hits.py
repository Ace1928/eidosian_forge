import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _write_hits(self, hits):
    """Write Hit objects (PRIVATE)."""
    xml = self.xml
    for num, hit in enumerate(hits):
        xml.startParent('Hit')
        xml.simpleElement('Hit_num', str(num + 1))
        opt_dict = {}
        if self._use_raw_hit_ids:
            hit_id = hit.blast_id
            hit_desc = ' >'.join([f'{x} {y}' for x, y in zip(hit.id_all, hit.description_all)])
        else:
            hit_id = hit.id
            hit_desc = hit.description + ' >'.join([f'{x} {y}' for x, y in zip(hit.id_all[1:], hit.description_all[1:])])
        opt_dict = {'Hit_id': hit_id, 'Hit_def': hit_desc}
        self._write_elem_block('Hit_', 'hit', hit, opt_dict)
        xml.startParent('Hit_hsps')
        self._write_hsps(hit.hsps)
        self.hit_counter += 1
        xml.endParents(2)