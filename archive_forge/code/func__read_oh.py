import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_oh(record, line):
    assert line[5:].startswith('NCBI_TaxID='), f'Unexpected {line}'
    line = line[16:].rstrip()
    assert line[-1] == '.' and line.count(';') == 1, line
    taxid, name = line[:-1].split(';')
    record.host_taxonomy_id.append(taxid.strip())
    record.host_organism.append(name.strip())