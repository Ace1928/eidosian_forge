import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
@staticmethod
def _feed_seq_length(consumer, text):
    length_parts = text.split()
    assert len(length_parts) == 2, f'Invalid sequence length string {text!r}'
    assert length_parts[1].upper() in ['BP', 'BP.', 'AA', 'AA.']
    consumer.size(length_parts[0])