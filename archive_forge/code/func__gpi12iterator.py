import copy
def _gpi12iterator(handle):
    """Read GPI 1.2 format files (PRIVATE).

    This iterator is used to read a gp_information.goa_uniprot
    file which is in the GPI 1.2 format.
    """
    for inline in handle:
        if inline[0] == '!':
            continue
        inrec = inline.rstrip('\n').split('\t')
        if len(inrec) == 1:
            continue
        inrec[3] = inrec[3].split('|')
        inrec[4] = inrec[4].split('|')
        inrec[8] = inrec[8].split('|')
        inrec[9] = inrec[9].split('|')
        yield dict(zip(GPI12FIELDS, inrec))