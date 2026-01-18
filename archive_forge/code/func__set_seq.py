import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _set_seq(self, seq, seq_type):
    """Check the given sequence for attribute setting (PRIVATE).

        :param seq: sequence to check
        :type seq: string or SeqRecord
        :param seq_type: sequence type
        :type seq_type: string, choice of 'hit' or 'query'

        """
    assert seq_type in ('hit', 'query')
    if seq is None:
        return seq
    elif not isinstance(seq, (str, SeqRecord)):
        raise TypeError('%s sequence must be a string or a SeqRecord object.' % seq_type)
    opp_type = 'hit' if seq_type == 'query' else 'query'
    opp_seq = getattr(self, '_%s' % opp_type, None)
    if opp_seq is not None:
        if len(seq) != len(opp_seq):
            raise ValueError('Sequence lengths do not match. Expected: %r (%s); found: %r (%s).' % (len(opp_seq), opp_type, len(seq), seq_type))
    seq_id = getattr(self, '%s_id' % seq_type)
    seq_desc = getattr(self, '%s_description' % seq_type)
    seq_feats = getattr(self, '%s_features' % seq_type)
    seq_name = 'aligned %s sequence' % seq_type
    if isinstance(seq, SeqRecord):
        seq.id = seq_id
        seq.description = seq_desc
        seq.name = seq_name
        seq.features = seq_feats
        seq.annotations['molecule_type'] = self.molecule_type
    elif isinstance(seq, str):
        seq = SeqRecord(Seq(seq), id=seq_id, name=seq_name, description=seq_desc, features=seq_feats, annotations={'molecule_type': self.molecule_type})
    return seq