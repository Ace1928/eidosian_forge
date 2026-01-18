from Bio import motifs
def _read_motif_name(handle):
    """Read motif name (PRIVATE)."""
    for line in handle:
        if 'sorted by position p-value' in line:
            break
    else:
        raise ValueError('Unexpected end of stream: Failed to find motif name')
    line = line.strip()
    words = line.split()
    name = ' '.join(words[0:2])
    return name