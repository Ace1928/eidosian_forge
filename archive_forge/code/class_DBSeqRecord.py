from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
class DBSeqRecord(SeqRecord):
    """BioSQL equivalent of the Biopython SeqRecord object."""

    def __init__(self, adaptor, primary_id):
        """Create a DBSeqRecord object.

        Arguments:
         - adaptor - A BioSQL.BioSeqDatabase.Adaptor object
         - primary_id - An internal integer ID used by BioSQL

        You wouldn't normally create a DBSeqRecord object yourself,
        this is done for you when using a BioSeqDatabase object
        """
        self._adaptor = adaptor
        self._primary_id = primary_id
        self._biodatabase_id, self._taxon_id, self.name, accession, version, self._identifier, self._division, self.description = self._adaptor.execute_one('SELECT biodatabase_id, taxon_id, name, accession, version, identifier, division, description FROM bioentry WHERE bioentry_id = %s', (self._primary_id,))
        if version and version != '0':
            self.id = f'{accession}.{version}'
        else:
            self.id = accession
        length = _retrieve_seq_len(adaptor, primary_id)
        self._per_letter_annotations = _RestrictedDict(length=length)

    def __get_seq(self):
        if not hasattr(self, '_seq'):
            self._seq = _retrieve_seq(self._adaptor, self._primary_id)
        return self._seq

    def __set_seq(self, seq):
        self._seq = seq

    def __del_seq(self):
        del self._seq
    seq = property(__get_seq, __set_seq, __del_seq, 'Seq object')

    @property
    def dbxrefs(self) -> List[str]:
        """Database cross references."""
        if not hasattr(self, '_dbxrefs'):
            self._dbxrefs = _retrieve_dbxrefs(self._adaptor, self._primary_id)
        return self._dbxrefs

    @dbxrefs.setter
    def dbxrefs(self, value: List[str]) -> None:
        self._dbxrefs = value

    @dbxrefs.deleter
    def dbxrefs(self) -> None:
        del self._dbxrefs

    def __get_features(self):
        if not hasattr(self, '_features'):
            self._features = _retrieve_features(self._adaptor, self._primary_id)
        return self._features

    def __set_features(self, features):
        self._features = features

    def __del_features(self):
        del self._features
    features = property(__get_features, __set_features, __del_features, 'Features')

    @property
    def annotations(self) -> SeqRecord._AnnotationsDict:
        """Annotations."""
        if not hasattr(self, '_annotations'):
            self._annotations = _retrieve_annotations(self._adaptor, self._primary_id, self._taxon_id)
            if self._identifier:
                self._annotations['gi'] = self._identifier
            if self._division:
                self._annotations['data_file_division'] = self._division
        return self._annotations

    @annotations.setter
    def annotations(self, value: Optional[SeqRecord._AnnotationsDict]) -> None:
        if value:
            self._annotations = value
        else:
            self._annotations = {}

    @annotations.deleter
    def annotations(self) -> None:
        del self._annotations