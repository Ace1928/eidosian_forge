import binascii
import struct
import sys
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
Write the complete file with the records, and return the number of records.