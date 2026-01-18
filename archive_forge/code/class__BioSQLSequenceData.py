from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
class _BioSQLSequenceData(SequenceDataAbstractBaseClass):
    """Retrieves sequence data from a BioSQL database (PRIVATE)."""
    __slots__ = ('primary_id', 'adaptor', '_length', 'start')

    def __init__(self, primary_id, adaptor, start=0, length=0):
        """Create a new _BioSQLSequenceData object referring to a BioSQL entry.

        You wouldn't normally create a _BioSQLSequenceData object yourself,
        this is done for you when retrieving a DBSeqRecord object from the
        database, which creates a Seq object using a _BioSQLSequenceData
        instance as the data provider.
        """
        self.primary_id = primary_id
        self.adaptor = adaptor
        self._length = length
        self.start = start
        super().__init__()

    def __len__(self):
        """Return the length of the sequence."""
        return self._length

    def __getitem__(self, key):
        """Return a subsequence as a bytes or a _BioSQLSequenceData object."""
        if isinstance(key, slice):
            start, end, step = key.indices(self._length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
        else:
            i = key
            if i < 0:
                i += self._length
                if i < 0:
                    raise IndexError(key)
            elif i >= self._length:
                raise IndexError(key)
            c = self.adaptor.get_subseq_as_string(self.primary_id, self.start + i, self.start + i + 1)
            return ord(c)
        if step == 1:
            if start == 0 and size == self._length:
                sequence = self.adaptor.get_subseq_as_string(self.primary_id, self.start, self.start + self._length)
                return sequence.encode('ASCII')
            else:
                return _BioSQLSequenceData(self.primary_id, self.adaptor, self.start + start, size)
        else:
            full = self.adaptor.get_subseq_as_string(self.primary_id, self.start + start, self.start + end)
            return full[::step].encode('ASCII')