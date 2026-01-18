from Bio import motifs
def _read_motif_statistics(handle):
    """Read motif statistics (PRIVATE)."""
    for line in handle:
        if line.startswith('letter-probability matrix:'):
            break
    num_occurrences = int(line.split('nsites=')[1].split()[0])
    length = int(line.split('w=')[1].split()[0])
    evalue = float(line.split('E=')[1].split()[0])
    return (length, num_occurrences, evalue)