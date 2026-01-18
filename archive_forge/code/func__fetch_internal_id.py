import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def _fetch_internal_id(self, base_id, version):
    """Fetch the internal id for a base id + version (PRIVATE).

        Also checks if this combo exists or not.
        """
    cur = self.dbh.cursor()
    cur.execute('select id from MATRIX where BASE_id = %s and VERSION = %s', (base_id, version))
    row = cur.fetchone()
    int_id = None
    if row:
        int_id = row[0]
    else:
        warnings.warn(f"Failed to fetch internal database ID for JASPAR motif with matrix ID '{base_id}.{version}'. No JASPAR motif with this matrix ID appears to exist.", BiopythonWarning)
    return int_id