import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def _feed_first_line_patents(self, consumer, line):
    fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip()[:-3].split(';')]
    assert len(fields) == 4
    consumer.locus(fields[0])
    consumer.residue_type(fields[1])
    consumer.data_file_division(fields[2])