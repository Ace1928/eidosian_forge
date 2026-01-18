from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_reference(adaptor, primary_id):
    refs = adaptor.execute_and_fetchall('SELECT start_pos, end_pos,  location, title, authors, dbname, accession FROM bioentry_reference JOIN reference USING (reference_id) LEFT JOIN dbxref USING (dbxref_id) WHERE bioentry_id = %s ORDER BY "rank"', (primary_id,))
    references = []
    for start, end, location, title, authors, dbname, accession in refs:
        reference = SeqFeature.Reference()
        if start is not None or end is not None:
            if start is not None:
                start -= 1
            reference.location = [SeqFeature.SimpleLocation(start, end)]
        if authors:
            reference.authors = authors
        if title:
            reference.title = title
        reference.journal = location
        if dbname == 'PUBMED':
            reference.pubmed_id = accession
        elif dbname == 'MEDLINE':
            reference.medline_id = accession
        references.append(reference)
    if references:
        return {'references': references}
    else:
        return {}