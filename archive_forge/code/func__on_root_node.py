from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _on_root_node(self, name):
    if name == 'BlastOutput':
        self._setup_blast_v1()
    elif name == 'BlastXML2':
        self._setup_blast_v2()
    else:
        raise ValueError('Invalid root node name: %s. Root node should be either BlastOutput or BlastXML2' % name)