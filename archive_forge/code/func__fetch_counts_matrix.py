import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def _fetch_counts_matrix(self, int_id):
    """Fetch the counts matrix from the JASPAR DB by the internal ID (PRIVATE).

        Returns a Bio.motifs.matrix.GenericPositionMatrix
        """
    counts = {}
    cur = self.dbh.cursor()
    for base in 'ACGT':
        base_counts = []
        cur.execute('select val from MATRIX_DATA where ID = %s and row = %s order by col', (int_id, base))
        rows = cur.fetchall()
        for row in rows:
            base_counts.append(row[0])
        counts[base] = [float(x) for x in base_counts]
    return matrix.GenericPositionMatrix('ACGT', counts)