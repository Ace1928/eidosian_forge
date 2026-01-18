from Bio import motifs
def _read_version(record, handle):
    """Read MEME version (PRIVATE)."""
    for line in handle:
        if line.startswith('MEME version'):
            break
    else:
        raise ValueError('Improper input file. File should contain a line starting MEME version.')
    line = line.strip()
    ls = line.split()
    record.version = ls[2]