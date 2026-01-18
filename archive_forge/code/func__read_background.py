from Bio import motifs
def _read_background(record, handle):
    """Read background letter frequencies (PRIVATE)."""
    for line in handle:
        if line.startswith('Background letter frequencies'):
            break
    else:
        raise ValueError('Improper input file. File should contain a line starting background frequencies.')
    try:
        line = next(handle)
    except StopIteration:
        raise ValueError('Unexpected end of stream: Expected to find line starting background frequencies.')
    line = line.strip()
    ls = line.split()
    A, C, G, T = (float(ls[1]), float(ls[3]), float(ls[5]), float(ls[7]))
    record.background = {'A': A, 'C': C, 'G': G, 'T': T}