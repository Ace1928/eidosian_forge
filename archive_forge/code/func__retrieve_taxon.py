from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature
def _retrieve_taxon(adaptor, primary_id, taxon_id):
    a = {}
    common_names = adaptor.execute_and_fetch_col0("SELECT name FROM taxon_name WHERE taxon_id = %s AND name_class = 'genbank common name'", (taxon_id,))
    if common_names:
        a['source'] = common_names[0]
    scientific_names = adaptor.execute_and_fetch_col0("SELECT name FROM taxon_name WHERE taxon_id = %s AND name_class = 'scientific name'", (taxon_id,))
    if scientific_names:
        a['organism'] = scientific_names[0]
    ncbi_taxids = adaptor.execute_and_fetch_col0('SELECT ncbi_taxon_id FROM taxon WHERE taxon_id = %s', (taxon_id,))
    if ncbi_taxids and ncbi_taxids[0] and (ncbi_taxids[0] != '0'):
        a['ncbi_taxid'] = ncbi_taxids[0]
    taxonomy = []
    while taxon_id:
        name, rank, parent_taxon_id = adaptor.execute_one("SELECT taxon_name.name, taxon.node_rank, taxon.parent_taxon_id FROM taxon, taxon_name WHERE taxon.taxon_id=taxon_name.taxon_id AND taxon_name.name_class='scientific name' AND taxon.taxon_id = %s", (taxon_id,))
        if taxon_id == parent_taxon_id:
            break
        taxonomy.insert(0, name)
        taxon_id = parent_taxon_id
    if taxonomy:
        a['taxonomy'] = taxonomy
    return a