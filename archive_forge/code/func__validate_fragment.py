import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _validate_fragment(self, fragment):
    if not isinstance(fragment, HSPFragment):
        raise TypeError('HSP objects can only contain HSPFragment objects.')
    if self._items:
        if fragment.hit_id != self.hit_id:
            raise ValueError('Expected HSPFragment with hit ID %r, found %r instead.' % (self.id, fragment.hit_id))
        if fragment.hit_description != self.hit_description:
            raise ValueError('Expected HSPFragment with hit description %r, found %r instead.' % (self.description, fragment.hit_description))
        if fragment.query_id != self.query_id:
            raise ValueError('Expected HSPFragment with query ID %r, found %r instead.' % (self.query_id, fragment.query_id))
        if fragment.query_description != self.query_description:
            raise ValueError('Expected HSP with query description %r, found %r instead.' % (self.query_description, fragment.query_description))