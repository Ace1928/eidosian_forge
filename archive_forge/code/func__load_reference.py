from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_reference(self, reference, rank, bioentry_id):
    """Record SeqRecord's annotated references in the database (PRIVATE).

        Arguments:
         - record - a SeqRecord object with annotated references
         - bioentry_id - corresponding database identifier

        """
    refs = None
    if reference.medline_id:
        refs = self.adaptor.execute_and_fetch_col0("SELECT reference_id FROM reference JOIN dbxref USING (dbxref_id) WHERE dbname = 'MEDLINE' AND accession = %s", (reference.medline_id,))
    if not refs and reference.pubmed_id:
        refs = self.adaptor.execute_and_fetch_col0("SELECT reference_id FROM reference JOIN dbxref USING (dbxref_id) WHERE dbname = 'PUBMED' AND accession = %s", (reference.pubmed_id,))
    if not refs:
        s = []
        for f in (reference.authors, reference.title, reference.journal):
            s.append(f or '<undef>')
        crc = crc64(''.join(s))
        refs = self.adaptor.execute_and_fetch_col0('SELECT reference_id FROM reference WHERE crc = %s', (crc,))
    if not refs:
        if reference.medline_id:
            dbxref_id = self._add_dbxref('MEDLINE', reference.medline_id, 0)
        elif reference.pubmed_id:
            dbxref_id = self._add_dbxref('PUBMED', reference.pubmed_id, 0)
        else:
            dbxref_id = None
        authors = reference.authors or None
        title = reference.title or None
        journal = reference.journal or ''
        self.adaptor.execute('INSERT INTO reference (dbxref_id, location, title, authors, crc) VALUES (%s, %s, %s, %s, %s)', (dbxref_id, journal, title, authors, crc))
        reference_id = self.adaptor.last_id('reference')
    else:
        reference_id = refs[0]
    if reference.location:
        start = 1 + int(str(reference.location[0].start))
        end = int(str(reference.location[0].end))
    else:
        start = None
        end = None
    sql = 'INSERT INTO bioentry_reference (bioentry_id, reference_id, start_pos, end_pos, "rank") VALUES (%s, %s, %s, %s, %s)'
    self.adaptor.execute(sql, (bioentry_id, reference_id, start, end, rank + 1))