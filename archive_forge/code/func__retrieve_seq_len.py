from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_seq_len(adaptor, primary_id):
    seqs = adaptor.execute_and_fetchall('SELECT length FROM biosequence WHERE bioentry_id = %s', (primary_id,))
    if not seqs:
        return None
    if len(seqs) != 1:
        raise ValueError(f'Expected 1 response, got {len(seqs)}.')
    given_length, = seqs[0]
    return int(given_length)