import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def _feed_first_line_patents_kipo(self, consumer, line):
    assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
    fields = [line[self.HEADER_WIDTH:].split(None, 1)[0]]
    fields.extend(line[self.HEADER_WIDTH:].split(None, 1)[1].split(';'))
    fields = [entry.strip() for entry in fields]
    "\n        The tokens represent:\n\n           0. Primary accession number\n           (space sep)\n           1. ??? (e.g. standard)\n           (semi-colon)\n           2. Molecule type (protein)? Division? Always 'PRT'\n           3. Sequence length (e.g. '111 AA.')\n        "
    consumer.locus(fields[0])
    self._feed_seq_length(consumer, fields[3])