from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_dbxrefs(self, record, bioentry_id):
    """Load any sequence level cross references into the database (PRIVATE).

        See table bioentry_dbxref.
        """
    for rank, value in enumerate(record.dbxrefs):
        newline_escape_count = value.count('\n')
        if newline_escape_count != 0:
            raise ValueError('Expected a single line in value, got {newline_escape_count}')
        try:
            db, accession = value.split(':', 1)
            db = db.strip()
            accession = accession.strip()
        except Exception:
            raise ValueError(f"Parsing of dbxrefs list failed: '{value}'") from None
        dbxref_id = self._get_dbxref_id(db, accession)
        self._get_bioentry_dbxref(bioentry_id, dbxref_id, rank + 1)