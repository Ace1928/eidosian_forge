class CannotReverseCherrypick(BzrError):
    _fmt = 'Selected merge cannot perform reverse cherrypicks.  Try merge3 or diff3.'