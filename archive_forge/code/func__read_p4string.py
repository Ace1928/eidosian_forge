from Textco BioSoftware, Inc.
from struct import unpack
from Bio.Seq import Seq
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _read_p4string(handle):
    """Read a 32-bit Pascal string.

    Similar to a Pascal string but length is encoded on 4 bytes.
    """
    length = _read(handle, 4)
    length = unpack('>I', length)[0]
    data = _read(handle, length).decode('ASCII')
    return data