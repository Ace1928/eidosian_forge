import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def _feed_first_line_old(self, consumer, line):
    assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
    fields = [line[self.HEADER_WIDTH:].split(None, 1)[0]]
    fields.extend(line[self.HEADER_WIDTH:].split(None, 1)[1].split(';'))
    fields = [entry.strip() for entry in fields]
    "\n        The tokens represent:\n\n           0. Primary accession number\n           (space sep)\n           1. ??? (e.g. standard)\n           (semi-colon)\n           2. Topology and/or Molecule type (e.g. 'circular DNA' or 'DNA')\n           3. Taxonomic division (e.g. 'PRO')\n           4. Sequence length (e.g. '4639675 BP.')\n\n        "
    consumer.locus(fields[0])
    consumer.residue_type(fields[2])
    if 'circular' in fields[2]:
        consumer.topology('circular')
        consumer.molecule_type(fields[2].replace('circular', '').strip())
    elif 'linear' in fields[2]:
        consumer.topology('linear')
        consumer.molecule_type(fields[2].replace('linear', '').strip())
    else:
        consumer.molecule_type(fields[2].strip())
    consumer.data_file_division(fields[3])
    self._feed_seq_length(consumer, fields[4])