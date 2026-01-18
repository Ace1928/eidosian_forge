from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_alphabet(adaptor, primary_id):
    results = adaptor.execute_and_fetchall('SELECT alphabet FROM biosequence WHERE bioentry_id = %s', (primary_id,))
    if len(results) != 1:
        raise ValueError(f'Expected 1 response, got {len(results)}.')
    alphabets = results[0]
    if len(alphabets) != 1:
        raise ValueError(f'Expected 1 alphabet in response, got {len(alphabets)}.')
    alphabet = alphabets[0]
    if alphabet == 'dna':
        molecule_type = 'DNA'
    elif alphabet == 'rna':
        molecule_type = 'RNA'
    elif alphabet == 'protein':
        molecule_type = 'protein'
    else:
        molecule_type = None
    if molecule_type is not None:
        return {'molecule_type': molecule_type}
    else:
        return {}