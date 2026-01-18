from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _get_taxon_id_from_ncbi_taxon_id(self, ncbi_taxon_id, scientific_name=None, common_name=None):
    """Get the taxon id for record from NCBI taxon ID (PRIVATE).

        Arguments:
         - ncbi_taxon_id - string containing an NCBI taxon id
         - scientific_name - string, used if a stub entry is recorded
         - common_name - string, used if a stub entry is recorded

        This searches the taxon table using ONLY the NCBI taxon ID
        to find the matching taxon table entry's ID (database key).

        If the species isn't in the taxon table, and the fetch_NCBI_taxonomy
        flag is true, Biopython will attempt to go online using Bio.Entrez
        to fetch the official NCBI lineage, recursing up the tree until an
        existing entry is found in the database or the full lineage has been
        fetched.

        Otherwise the NCBI taxon ID, scientific name and common name are
        recorded as a minimal stub entry in the taxon and taxon_name tables.
        Any partial information about the lineage from the SeqRecord is NOT
        recorded.  This should mean that (re)running the BioSQL script
        load_ncbi_taxonomy.pl can fill in the taxonomy lineage.

        Returns the taxon id (database key for the taxon table, not
        an NCBI taxon ID).
        """
    if not ncbi_taxon_id:
        raise ValueError('Expected a non-empty value for ncbi_taxon_id.')
    taxon_id = self.adaptor.execute_and_fetch_col0('SELECT taxon_id FROM taxon WHERE ncbi_taxon_id = %s', (int(ncbi_taxon_id),))
    if taxon_id:
        return taxon_id[0]
    parent_taxon_id = None
    rank = 'species'
    genetic_code = None
    mito_genetic_code = None
    parent_left_value = None
    parent_right_value = None
    left_value = None
    right_value = None
    species_names = []
    if scientific_name:
        species_names.append(('scientific name', scientific_name))
    if common_name:
        species_names.append(('common name', common_name))
    if self.fetch_NCBI_taxonomy:
        handle = Entrez.efetch(db='taxonomy', id=ncbi_taxon_id, retmode='XML')
        taxonomic_record = Entrez.read(handle)
        if len(taxonomic_record) == 1:
            if taxonomic_record[0]['TaxId'] != str(ncbi_taxon_id):
                raise ValueError(f'ncbi_taxon_id different from parent taxon id. {ncbi_taxon_id} versus {taxonomic_record[0]['TaxId']}')
            parent_taxon_id, parent_left_value, parent_right_value = self._get_taxon_id_from_ncbi_lineage(taxonomic_record[0]['LineageEx'])
            left_value = parent_right_value
            right_value = parent_right_value + 1
            rank = str(taxonomic_record[0]['Rank'])
            genetic_code = int(taxonomic_record[0]['GeneticCode']['GCId'])
            mito_genetic_code = int(taxonomic_record[0]['MitoGeneticCode']['MGCId'])
            species_names = [('scientific name', str(taxonomic_record[0]['ScientificName']))]
            try:
                for name_class, names in taxonomic_record[0]['OtherNames'].items():
                    name_class = self._fix_name_class(name_class)
                    if not isinstance(names, list):
                        names = [names]
                    for name in names:
                        if isinstance(name, str):
                            species_names.append((name_class, name))
            except KeyError:
                pass
    else:
        pass
    self._update_left_right_taxon_values(left_value)
    self.adaptor.execute('INSERT INTO taxon(parent_taxon_id, ncbi_taxon_id, node_rank, genetic_code, mito_genetic_code, left_value, right_value) VALUES (%s, %s, %s, %s, %s, %s, %s)', (parent_taxon_id, ncbi_taxon_id, rank, genetic_code, mito_genetic_code, left_value, right_value))
    taxon_id = self.adaptor.last_id('taxon')
    for name_class, name in species_names:
        self.adaptor.execute('INSERT INTO taxon_name(taxon_id, name, name_class) VALUES (%s, %s, %s)', (taxon_id, name[:255], name_class))
    return taxon_id