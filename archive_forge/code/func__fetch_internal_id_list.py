import warnings
from Bio import BiopythonWarning
from Bio import MissingPythonDependencyError
from Bio.motifs import jaspar, matrix
def _fetch_internal_id_list(self, collection=JASPAR_DFLT_COLLECTION, tf_name=None, tf_class=None, tf_family=None, matrix_id=None, tax_group=None, species=None, pazar_id=None, data_type=None, medline=None, all=False, all_versions=False):
    """Fetch list of internal JASPAR motif IDs.

        Fetch a list of internal JASPAR motif IDs based on various passed
        parameters which may then be used to fetch the rest of the motif data.

        Caller:
            fetch_motifs()

        Arguments:
            See arguments sections of fetch_motifs()

        Returns:
            A list of internal JASPAR motif IDs which match the given
            selection criteria arguments.


        Build an SQL query based on the selection arguments provided.

        1: First add table joins and sub-clauses for criteria corresponding to
           named fields from the MATRIX and MATRIX_SPECIES tables such as
           collection, matrix ID, name, species etc.

        2: Then add joins/sub-clauses for tag/value parameters from the
           MATRIX_ANNOTATION table.

        For the surviving matrices, the responsibility to do matrix-based
        feature filtering such as ic, number of sites etc, fall on the
        calling fetch_motifs() method.

        """
    int_ids = []
    cur = self.dbh.cursor()
    '\n        Special case 1: fetch ALL motifs. Highest priority.\n        Ignore all other selection arguments.\n        '
    if all:
        cur.execute('select ID from MATRIX')
        rows = cur.fetchall()
        for row in rows:
            int_ids.append(row[0])
        return int_ids
    "\n        Special case 2: fetch specific motifs by their JASPAR IDs. This\n        has higher priority than any other except the above 'all' case.\n        Ignore all other selection arguments.\n        "
    if matrix_id:
        '\n            These might be either stable IDs or stable_ID.version.\n            If just stable ID and if all_versions == 1, return all versions,\n            otherwise just the latest\n            '
        if all_versions:
            for id in matrix_id:
                base_id, version = jaspar.split_jaspar_id(id)
                cur.execute('select ID from MATRIX where BASE_ID = %s', (base_id,))
                rows = cur.fetchall()
                for row in rows:
                    int_ids.append(row[0])
        else:
            for id in matrix_id:
                base_id, version = jaspar.split_jaspar_id(id)
                if not version:
                    version = self._fetch_latest_version(base_id)
                int_id = None
                if version:
                    int_id = self._fetch_internal_id(base_id, version)
                if int_id:
                    int_ids.append(int_id)
        return int_ids
    tables = ['MATRIX m']
    where_clauses = []
    if collection:
        if isinstance(collection, list):
            clause = "m.COLLECTION in ('"
            clause = ''.join([clause, "','".join(collection)])
            clause = ''.join([clause, "')"])
        else:
            clause = "m.COLLECTION = '%s'" % collection
        where_clauses.append(clause)
    if tf_name:
        if isinstance(tf_name, list):
            clause = "m.NAME in ('"
            clause = ''.join([clause, "','".join(tf_name)])
            clause = ''.join([clause, "')"])
        else:
            clause = "m.NAME = '%s'" % tf_name
        where_clauses.append(clause)
    if species:
        tables.append('MATRIX_SPECIES ms')
        where_clauses.append('m.ID = ms.ID')
        '\n            NOTE: species are numeric taxonomy IDs but stored as varchars\n            in the DB.\n            '
        if isinstance(species, list):
            clause = "ms.TAX_ID in ('"
            clause = ''.join([clause, "','".join((str(s) for s in species))])
            clause = ''.join([clause, "')"])
        else:
            clause = "ms.TAX_ID = '%s'" % species
        where_clauses.append(clause)
    '\n        Tag based selection from MATRIX_ANNOTATION\n        Differs from perl TFBS module in that the matrix class explicitly\n        has a tag attribute corresponding to the tags in the database. This\n        provides tremendous flexibility in adding new tags to the DB and\n        being able to select based on those tags with out adding new code.\n        In the JASPAR Motif class we have elected to use specific attributes\n        for the most commonly used tags and here correspondingly only allow\n        selection on these attributes.\n\n        The attributes corresponding to the tags for which selection is\n        provided are:\n\n           Attribute   Tag\n           tf_class    class\n           tf_family   family\n           pazar_id    pazar_tf_id\n           medline     medline\n           data_type   type\n           tax_group   tax_group\n        '
    if tf_class:
        tables.append('MATRIX_ANNOTATION ma1')
        where_clauses.append('m.ID = ma1.ID')
        clause = "ma1.TAG = 'class'"
        if isinstance(tf_class, list):
            clause = ''.join([clause, " and ma1.VAL in ('"])
            clause = ''.join([clause, "','".join(tf_class)])
            clause = ''.join([clause, "')"])
        else:
            clause = ''.join([clause, " and ma1.VAL = '%s' " % tf_class])
        where_clauses.append(clause)
    if tf_family:
        tables.append('MATRIX_ANNOTATION ma2')
        where_clauses.append('m.ID = ma2.ID')
        clause = "ma2.TAG = 'family'"
        if isinstance(tf_family, list):
            clause = ''.join([clause, " and ma2.VAL in ('"])
            clause = ''.join([clause, "','".join(tf_family)])
            clause = ''.join([clause, "')"])
        else:
            clause = ''.join([clause, " and ma2.VAL = '%s' " % tf_family])
        where_clauses.append(clause)
    if pazar_id:
        tables.append('MATRIX_ANNOTATION ma3')
        where_clauses.append('m.ID = ma3.ID')
        clause = "ma3.TAG = 'pazar_tf_id'"
        if isinstance(pazar_id, list):
            clause = ''.join([clause, " and ma3.VAL in ('"])
            clause = ''.join([clause, "','".join(pazar_id)])
            clause = ''.join([clause, "')"])
        else:
            clause = ''.join([" and ma3.VAL = '%s' " % pazar_id])
        where_clauses.append(clause)
    if medline:
        tables.append('MATRIX_ANNOTATION ma4')
        where_clauses.append('m.ID = ma4.ID')
        clause = "ma4.TAG = 'medline'"
        if isinstance(medline, list):
            clause = ''.join([clause, " and ma4.VAL in ('"])
            clause = ''.join([clause, "','".join(medline)])
            clause = ''.join([clause, "')"])
        else:
            clause = ''.join([" and ma4.VAL = '%s' " % medline])
        where_clauses.append(clause)
    if data_type:
        tables.append('MATRIX_ANNOTATION ma5')
        where_clauses.append('m.ID = ma5.ID')
        clause = "ma5.TAG = 'type'"
        if isinstance(data_type, list):
            clause = ''.join([clause, " and ma5.VAL in ('"])
            clause = ''.join([clause, "','".join(data_type)])
            clause = ''.join([clause, "')"])
        else:
            clause = ''.join([" and ma5.VAL = '%s' " % data_type])
        where_clauses.append(clause)
    if tax_group:
        tables.append('MATRIX_ANNOTATION ma6')
        where_clauses.append('m.ID = ma6.ID')
        clause = "ma6.TAG = 'tax_group'"
        if isinstance(tax_group, list):
            clause = ''.join([clause, " and ma6.VAL in ('"])
            clause = ''.join([clause, "','".join(tax_group)])
            clause = ''.join([clause, "')"])
        else:
            clause = ''.join([clause, " and ma6.VAL = '%s' " % tax_group])
        where_clauses.append(clause)
    sql = ''.join(['select distinct(m.ID) from ', ', '.join(tables)])
    if where_clauses:
        sql = ''.join([sql, ' where ', ' and '.join(where_clauses)])
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        id = row[0]
        if all_versions:
            int_ids.append(id)
        elif self._is_latest_version(id):
            int_ids.append(id)
    if len(int_ids) < 1:
        warnings.warn('Zero motifs returned with current select criteria', BiopythonWarning)
    return int_ids