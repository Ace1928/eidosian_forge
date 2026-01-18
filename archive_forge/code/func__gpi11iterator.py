import copy
def _gpi11iterator(handle):
    """Read GPI 1.1 format files (PRIVATE).

    This iterator is used to read a gp_information.goa_uniprot
    file which is in the GPI 1.1 format.
    """
    for inline in handle:
        if inline[0] == '!':
            continue
        inrec = inline.rstrip('\n').split('\t')
        if len(inrec) == 1:
            continue
        inrec[2] = inrec[2].split('|')
        inrec[3] = inrec[3].split('|')
        inrec[7] = inrec[7].split('|')
        inrec[8] = inrec[8].split('|')
        yield dict(zip(GPI11FIELDS, inrec))