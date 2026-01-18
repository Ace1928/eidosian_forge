from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_dbxrefs(adaptor, primary_id):
    """Retrieve the database cross references for the sequence (PRIVATE)."""
    _dbxrefs = []
    dbxrefs = adaptor.execute_and_fetchall('SELECT dbname, accession, version FROM bioentry_dbxref join dbxref using (dbxref_id) WHERE bioentry_id = %s ORDER BY "rank"', (primary_id,))
    for dbname, accession, version in dbxrefs:
        if version and version != '0':
            v = f'{accession}.{version}'
        else:
            v = accession
        _dbxrefs.append(f'{dbname}:{v}')
    return _dbxrefs