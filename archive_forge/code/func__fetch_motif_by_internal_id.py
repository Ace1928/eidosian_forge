import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def _fetch_motif_by_internal_id(self, int_id):
    """Fetch basic motif information (PRIVATE)."""
    cur = self.dbh.cursor()
    cur.execute('select BASE_ID, VERSION, COLLECTION, NAME from MATRIX where id = %s', (int_id,))
    row = cur.fetchone()
    if not row:
        warnings.warn(f'Could not fetch JASPAR motif with internal ID = {int_id}', BiopythonWarning)
        return None
    base_id = row[0]
    version = row[1]
    collection = row[2]
    name = row[3]
    matrix_id = ''.join([base_id, '.', str(version)])
    counts = self._fetch_counts_matrix(int_id)
    motif = jaspar.Motif(matrix_id, name, collection=collection, counts=counts)
    cur.execute('select TAX_ID from MATRIX_SPECIES where id = %s', (int_id,))
    tax_ids = []
    rows = cur.fetchall()
    for row in rows:
        tax_ids.append(row[0])
    motif.species = tax_ids
    cur.execute('select ACC FROM MATRIX_PROTEIN where id = %s', (int_id,))
    accs = []
    rows = cur.fetchall()
    for row in rows:
        accs.append(row[0])
    motif.acc = accs
    cur.execute('select TAG, VAL from MATRIX_ANNOTATION where id = %s', (int_id,))
    rows = cur.fetchall()
    tf_family = []
    tf_class = []
    for row in rows:
        attr = row[0]
        val = row[1]
        if attr == 'class':
            tf_class.append(val)
        elif attr == 'family':
            tf_family.append(val)
        elif attr == 'tax_group':
            motif.tax_group = val
        elif attr == 'type':
            motif.data_type = val
        elif attr == 'pazar_tf_id':
            motif.pazar_id = val
        elif attr == 'medline':
            motif.medline = val
        elif attr == 'comment':
            motif.comment = val
        else:
            pass
    motif.tf_family = tf_family
    motif.tf_class = tf_class
    return motif