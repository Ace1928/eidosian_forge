class BzrBadParameterUnicode(BzrBadParameter):
    _fmt = 'Parameter %(param)s is unicode but only byte-strings are permitted.'