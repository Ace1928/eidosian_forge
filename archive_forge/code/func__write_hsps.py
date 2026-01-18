import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _write_hsps(self, hsps):
    """Write HSP objects (PRIVATE)."""
    xml = self.xml
    for num, hsp in enumerate(hsps):
        xml.startParent('Hsp')
        xml.simpleElement('Hsp_num', str(num + 1))
        for elem, attr in _WRITE_MAPS['hsp']:
            elem = 'Hsp_' + elem
            try:
                content = self._adjust_output(hsp, elem, attr)
            except AttributeError:
                if elem not in _DTD_OPT:
                    raise ValueError(f'Element {elem} (attribute {attr}) not found')
            else:
                xml.simpleElement(elem, str(content))
        self.hsp_counter += 1
        self.frag_counter += len(hsp.fragments)
        xml.endParent()