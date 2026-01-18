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
def _write_sequence(self, record):
    handle = self.handle
    try:
        data = _get_seq_string(record)
    except UndefinedSequenceError:
        if 'contig' in record.annotations:
            self._write_contig(record)
        else:
            handle.write('SQ   \n')
        return
    data = data.lower()
    seq_len = len(data)
    molecule_type = record.annotations.get('molecule_type')
    if molecule_type is not None and 'DNA' in molecule_type:
        a_count = data.count('A') + data.count('a')
        c_count = data.count('C') + data.count('c')
        g_count = data.count('G') + data.count('g')
        t_count = data.count('T') + data.count('t')
        other = seq_len - (a_count + c_count + g_count + t_count)
        handle.write('SQ   Sequence %i BP; %i A; %i C; %i G; %i T; %i other;\n' % (seq_len, a_count, c_count, g_count, t_count, other))
    else:
        handle.write('SQ   \n')
    for line_number in range(seq_len // self.LETTERS_PER_LINE):
        handle.write('    ')
        for block in range(self.BLOCKS_PER_LINE):
            index = self.LETTERS_PER_LINE * line_number + self.LETTERS_PER_BLOCK * block
            handle.write(f' {data[index:index + self.LETTERS_PER_BLOCK]}')
        handle.write(str((line_number + 1) * self.LETTERS_PER_LINE).rjust(self.POSITION_PADDING))
        handle.write('\n')
    if seq_len % self.LETTERS_PER_LINE:
        line_number = seq_len // self.LETTERS_PER_LINE
        handle.write('    ')
        for block in range(self.BLOCKS_PER_LINE):
            index = self.LETTERS_PER_LINE * line_number + self.LETTERS_PER_BLOCK * block
            handle.write(f' {data[index:index + self.LETTERS_PER_BLOCK]}'.ljust(11))
        handle.write(str(seq_len).rjust(self.POSITION_PADDING))
        handle.write('\n')