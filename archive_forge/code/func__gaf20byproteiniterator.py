import copy
def _gaf20byproteiniterator(handle):
    cur_id = None
    id_rec_list = []
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
        cur_rec = dict(zip(GAF20FIELDS, inrec))
        if cur_rec['DB_Object_ID'] != cur_id and cur_id:
            ret_list = copy.copy(id_rec_list)
            id_rec_list = [cur_rec]
            cur_id = cur_rec['DB_Object_ID']
            yield ret_list
        else:
            cur_id = cur_rec['DB_Object_ID']
            id_rec_list.append(cur_rec)