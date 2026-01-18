import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
def _read_gn(record):
    for i, text in enumerate(record.gene_name):
        tokens = text.rstrip('; ').split('; ')
        gene_name = {}
        for token in tokens:
            key, value = token.strip().split('=', 1)
            if key == 'Name':
                gene_name['Name'] = value
            else:
                assert key in ('Synonyms', 'OrderedLocusNames', 'ORFNames')
                gene_name[key] = value.split(', ')
        record.gene_name[i] = gene_name