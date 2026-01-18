import binascii
def _init_table_h():
    _table_h = []
    for i in range(256):
        part_l = i
        part_h = 0
        for j in range(8):
            rflag = part_l & 1
            part_l >>= 1
            if part_h & 1:
                part_l |= 1 << 31
            part_h >>= 1
            if rflag:
                part_h ^= 3623878656
        _table_h.append(part_h)
    return _table_h