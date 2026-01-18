from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_seqfeature(self, feature, feature_rank, bioentry_id):
    """Load a biopython SeqFeature into the database (PRIVATE)."""
    try:
        source = feature.qualifiers['source']
        if isinstance(source, list):
            source = source[0]
        seqfeature_id = self._load_seqfeature_basic(feature.type, feature_rank, bioentry_id, source=source)
    except KeyError:
        seqfeature_id = self._load_seqfeature_basic(feature.type, feature_rank, bioentry_id)
    self._load_seqfeature_locations(feature, seqfeature_id)
    self._load_seqfeature_qualifiers(feature.qualifiers, seqfeature_id)