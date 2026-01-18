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
def _parse_cookie_packet(length, data, record):
    """Parse a SnapGene cookie packet.

    Every SnapGene file starts with a packet of this type. It acts as
    a magic cookie identifying the file as a SnapGene file.
    """
    cookie, seq_type, exp_version, imp_version = unpack('>8sHHH', data)
    if cookie.decode('ASCII') != 'SnapGene':
        raise ValueError('The file is not a valid SnapGene file')