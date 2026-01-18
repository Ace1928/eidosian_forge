import copy
def _gaf10iterator(handle):
    for inline in handle:
        if inline[0] == '!':
            continue
        inrec = inline.rstrip('\n').split('\t')
        if len(inrec) == 1:
            continue
        inrec[3] = inrec[3].split('|')
        inrec[5] = inrec[5].split('|')
        inrec[7] = inrec[7].split('|')
        inrec[10] = inrec[10].split('|')
        inrec[12] = inrec[12].split('|')
        yield dict(zip(GAF10FIELDS, inrec))