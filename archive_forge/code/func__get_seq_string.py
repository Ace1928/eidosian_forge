from abc import ABC
from abc import abstractmethod
from os import PathLike
from typing import Iterator, IO, Optional, Union, Generic, AnyStr
from Bio import StreamModeError
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def _get_seq_string(record: SeqRecord) -> str:
    """Use this to catch errors like the sequence being None (PRIVATE)."""
    if not isinstance(record, SeqRecord):
        raise TypeError('Expected a SeqRecord object')
    if record.seq is None:
        raise TypeError(f'SeqRecord (id={record.id}) has None for its sequence.')
    elif not isinstance(record.seq, (Seq, MutableSeq)):
        raise TypeError(f'SeqRecord (id={record.id}) has an invalid sequence.')
    return str(record.seq)