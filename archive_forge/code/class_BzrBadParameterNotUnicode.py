class BzrBadParameterNotUnicode(BzrBadParameter):
    _fmt = 'Parameter %(param)s is neither unicode nor utf8.'