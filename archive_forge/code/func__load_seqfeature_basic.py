from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_seqfeature_basic(self, feature_type, feature_rank, bioentry_id, source='EMBL/GenBank/SwissProt'):
    """Load the first tables of a seqfeature and returns the id (PRIVATE).

        This loads the "key" of the seqfeature (ie. CDS, gene) and
        the basic seqfeature table itself.
        """
    ontology_id = self._get_ontology_id('SeqFeature Keys')
    seqfeature_key_id = self._get_term_id(feature_type, ontology_id=ontology_id)
    source_cat_id = self._get_ontology_id('SeqFeature Sources')
    source_term_id = self._get_term_id(source, ontology_id=source_cat_id)
    sql = 'INSERT INTO seqfeature (bioentry_id, type_term_id, source_term_id, "rank") VALUES (%s, %s, %s, %s)'
    self.adaptor.execute(sql, (bioentry_id, seqfeature_key_id, source_term_id, feature_rank + 1))
    return self.adaptor.last_id('seqfeature')