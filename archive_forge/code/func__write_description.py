from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_description(self, record):
    """Write the description if given (PRIVATE)."""
    if record.description:
        if not isinstance(record.description, str):
            raise TypeError('Description should be of type string')
        description = record.description
        if description == '<unknown description>':
            description = ''
        if len(record.description) > 0:
            self.xml_generator.startElement('description', AttributesImpl({}))
            self.xml_generator.characters(description)
            self.xml_generator.endElement('description')