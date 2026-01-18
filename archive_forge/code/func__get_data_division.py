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
@staticmethod
def _get_data_division(record):
    try:
        division = record.annotations['data_file_division']
    except KeyError:
        division = 'UNC'
    if division in ['PHG', 'ENV', 'FUN', 'HUM', 'INV', 'MAM', 'VRT', 'MUS', 'PLN', 'PRO', 'ROD', 'SYN', 'TGN', 'UNC', 'VRL', 'XXX']:
        pass
    else:
        gbk_to_embl = {'BCT': 'PRO', 'UNK': 'UNC'}
        try:
            division = gbk_to_embl[division]
        except KeyError:
            division = 'UNC'
    assert len(division) == 3
    return division