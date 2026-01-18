import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _write_references(self, record):
    number = 0
    for ref in record.annotations['references']:
        if not isinstance(ref, SeqFeature.Reference):
            continue
        number += 1
        self._write_single_line('RN', '[%i]' % number)
        if ref.location and len(ref.location) == 1:
            self._write_single_line('RP', '%i-%i' % (ref.location[0].start + 1, ref.location[0].end))
        if ref.pubmed_id:
            self._write_single_line('RX', f'PUBMED; {ref.pubmed_id}.')
        if ref.consrtm:
            self._write_single_line('RG', f'{ref.consrtm}')
        if ref.authors:
            self._write_multi_line('RA', ref.authors + ';')
        if ref.title:
            self._write_multi_line('RT', f'"{ref.title}";')
        if ref.journal:
            self._write_multi_line('RL', ref.journal)
        self.handle.write('XX\n')