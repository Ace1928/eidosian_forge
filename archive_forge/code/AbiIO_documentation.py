import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
Parse the file and generate SeqRecord objects.