from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_taxon_id_from_ncbi_lineage(self, taxonomic_lineage):
    """Recursive method to get taxon ID from NCBI lineage (PRIVATE).

        Arguments:
         - taxonomic_lineage - list of taxonomy dictionaries from Bio.Entrez

        First dictionary in list is the taxonomy root, highest would be
        the species. Each dictionary includes:

        - TaxID (string, NCBI taxon id)
        - Rank (string, e.g. "species", "genus", ..., "phylum", ...)
        - ScientificName (string)

        (and that is all at the time of writing)

        This method will record all the lineage given, returning the taxon id
        (database key, not NCBI taxon id) of the final entry (the species).
        """
    ncbi_taxon_id = int(taxonomic_lineage[-1]['TaxId'])
    left_value = None
    right_value = None
    parent_left_value = None
    parent_right_value = None
    rows = self.adaptor.execute_and_fetchall('SELECT taxon_id, left_value, right_value FROM taxon WHERE ncbi_taxon_id=%s' % ncbi_taxon_id)
    if rows:
        if len(rows) != 1:
            raise ValueError(f'Expected 1 reponse, got {len(rows)}')
        return rows[0]
    if len(taxonomic_lineage) > 1:
        parent_taxon_id, parent_left_value, parent_right_value = self._get_taxon_id_from_ncbi_lineage(taxonomic_lineage[:-1])
        left_value = parent_right_value
        right_value = parent_right_value + 1
        if not isinstance(parent_taxon_id, int):
            raise ValueError(f'Expected parent_taxon_id to be an int, got {parent_taxon_id}')
    else:
        parent_taxon_id = None
        left_value = self.adaptor.execute_one('SELECT MAX(left_value) FROM taxon')[0]
        if not left_value:
            left_value = 0
        right_value = left_value + 1
    self._update_left_right_taxon_values(left_value)
    rank = str(taxonomic_lineage[-1].get('Rank'))
    self.adaptor.execute('INSERT INTO taxon(ncbi_taxon_id, parent_taxon_id, node_rank, left_value, right_value) VALUES (%s, %s, %s, %s, %s)', (ncbi_taxon_id, parent_taxon_id, rank, left_value, right_value))
    taxon_id = self.adaptor.last_id('taxon')
    scientific_name = taxonomic_lineage[-1].get('ScientificName')
    if scientific_name:
        self.adaptor.execute("INSERT INTO taxon_name(taxon_id, name, name_class) VALUES (%s, %s, 'scientific name')", (taxon_id, scientific_name[:255]))
    return (taxon_id, left_value, right_value)