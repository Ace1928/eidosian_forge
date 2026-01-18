import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def _feed_first_line_new(self, consumer, line):
    assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
    fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip().split(';')]
    assert len(fields) == 7
    "\n        The tokens represent:\n\n           0. Primary accession number\n           1. Sequence version number\n           2. Topology: 'circular' or 'linear'\n           3. Molecule type (e.g. 'genomic DNA')\n           4. Data class (e.g. 'STD')\n           5. Taxonomic division (e.g. 'PRO')\n           6. Sequence length (e.g. '4639675 BP.')\n\n        "
    consumer.locus(fields[0])
    consumer.accession(fields[0])
    version_parts = fields[1].split()
    if len(version_parts) == 2 and version_parts[0] == 'SV' and version_parts[1].isdigit():
        consumer.version_suffix(version_parts[1])
    consumer.residue_type(' '.join(fields[2:4]))
    consumer.topology(fields[2])
    consumer.molecule_type(fields[3])
    consumer.data_file_division(fields[5])
    self._feed_seq_length(consumer, fields[6])