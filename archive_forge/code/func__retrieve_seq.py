from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_seq(adaptor, primary_id):
    seqs = adaptor.execute_and_fetchall('SELECT alphabet, length, length(seq) FROM biosequence WHERE bioentry_id = %s', (primary_id,))
    if not seqs:
        return
    if len(seqs) != 1:
        raise ValueError(f'Expected 1 response, got {len(seqs)}.')
    moltype, given_length, length = seqs[0]
    try:
        length = int(length)
        given_length = int(given_length)
        if length != given_length:
            raise ValueError(f"'length' differs from sequence length, {given_length}, {length}")
        have_seq = True
    except TypeError:
        if length is not None:
            raise ValueError(f"Expected 'length' to be 'None', got {length}.")
        seqs = adaptor.execute_and_fetchall('SELECT alphabet, length, seq FROM biosequence WHERE bioentry_id = %s', (primary_id,))
        if len(seqs) != 1:
            raise ValueError(f'Expected 1 response, got {len(seqs)}.')
        moltype, given_length, seq = seqs[0]
        if seq:
            raise ValueError(f"Expected 'seq' to have a falsy value, got {seq}.")
        length = int(given_length)
        have_seq = False
        del seq
    del given_length
    if have_seq:
        data = _BioSQLSequenceData(primary_id, adaptor, start=0, length=length)
        return Seq(data)
    else:
        return Seq(None, length=length)