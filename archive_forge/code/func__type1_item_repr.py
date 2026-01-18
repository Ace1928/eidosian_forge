from fontTools.encodings.StandardEncoding import StandardEncoding
def _type1_item_repr(key, value):
    psstring = ''
    access = _accessstrings[value.access]
    if access:
        access = access + ' '
    if key == 'CharStrings':
        psstring = psstring + '/%s %s def\n' % (key, _type1_CharString_repr(value.value))
    elif key == 'Encoding':
        psstring = psstring + _type1_Encoding_repr(value, access)
    else:
        psstring = psstring + '/%s %s %sdef\n' % (str(key), str(value), access)
    return psstring