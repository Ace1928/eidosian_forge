from GSL Biotech LLC.
from datetime import datetime
from re import sub
from struct import unpack
from xml.dom.minidom import parseString
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _parse_notes_packet(length, data, record):
    """Parse a 'Notes' packet.

    This type of packet contains some metadata about the sequence. They
    are stored as a XML string with a 'Notes' root node.
    """
    xml = parseString(data.decode('UTF-8'))
    type = _get_child_value(xml, 'Type')
    if type == 'Synthetic':
        record.annotations['data_file_division'] = 'SYN'
    else:
        record.annotations['data_file_division'] = 'UNC'
    date = _get_child_value(xml, 'LastModified')
    if date:
        record.annotations['date'] = datetime.strptime(date, '%Y.%m.%d')
    acc = _get_child_value(xml, 'AccessionNumber')
    if acc:
        record.id = acc
    comment = _get_child_value(xml, 'Comments')
    if comment:
        record.name = comment.split(' ', 1)[0]
        record.description = comment
        if not acc:
            record.id = record.name