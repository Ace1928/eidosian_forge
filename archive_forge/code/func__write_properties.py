from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_properties(self, record):
    """Write all annotations that are key value pairs with values of a primitive type or list of primitive types (PRIVATE)."""
    for key, value in record.annotations.items():
        if key not in ('organism', 'ncbi_taxid', 'source'):
            if value is None:
                attr = {'name': key}
                self.xml_generator.startElement('property', AttributesImpl(attr))
                self.xml_generator.endElement('property')
            elif isinstance(value, list):
                for v in value:
                    if v is None:
                        attr = {'name': key}
                    else:
                        attr = {'name': key, 'value': str(v)}
                    self.xml_generator.startElement('property', AttributesImpl(attr))
                    self.xml_generator.endElement('property')
            elif isinstance(value, (int, float, str)):
                attr = {'name': key, 'value': str(value)}
                self.xml_generator.startElement('property', AttributesImpl(attr))
                self.xml_generator.endElement('property')