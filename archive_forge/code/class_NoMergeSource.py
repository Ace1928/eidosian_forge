class NoMergeSource(BzrError):
    """Raise if no merge source was specified for a merge directive"""
    _fmt = 'A merge directive must provide either a bundle or a public branch location.'