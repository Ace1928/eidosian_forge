from Bio import motifs
def _read_alphabet(record, handle):
    """Read alphabet (PRIVATE)."""
    for line in handle:
        if line.startswith('ALPHABET'):
            break
    else:
        raise ValueError("Unexpected end of stream: Expected to find line starting with 'ALPHABET'")
    if not line.startswith('ALPHABET= '):
        raise ValueError("Line does not start with 'ALPHABET':\n%s" % line)
    line = line.strip().replace('ALPHABET= ', '')
    if line == 'ACGT':
        al = 'ACGT'
    else:
        al = 'ACDEFGHIKLMNPQRSTVWY'
    record.alphabet = al