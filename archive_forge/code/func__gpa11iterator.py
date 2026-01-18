import copy
def _gpa11iterator(handle):
    """Read GPA 1.1 format files (PRIVATE).

    This iterator is used to read a gp_association.goa_uniprot
    file which is in the GPA 1.1 format. Do not call directly. Rather
    use the gpa_iterator function
    """
    for inline in handle:
        if inline[0] == '!':
            continue
        inrec = inline.rstrip('\n').split('\t')
        if len(inrec) == 1:
            continue
        inrec[2] = inrec[2].split('|')
        inrec[4] = inrec[4].split('|')
        inrec[6] = inrec[6].split('|')
        inrec[10] = inrec[10].split('|')
        yield dict(zip(GPA11FIELDS, inrec))