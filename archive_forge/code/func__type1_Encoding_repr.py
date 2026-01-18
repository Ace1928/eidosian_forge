from fontTools.encodings.StandardEncoding import StandardEncoding
def _type1_Encoding_repr(encoding, access):
    encoding = encoding.value
    psstring = '/Encoding 256 array\n0 1 255 {1 index exch /.notdef put} for\n'
    for i in range(256):
        name = encoding[i].value
        if name != '.notdef':
            psstring = psstring + 'dup %d /%s put\n' % (i, name)
    return psstring + access + 'def\n'