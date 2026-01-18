class UnsupportedVcs(UnsupportedFormatError):
    vcs: str
    _fmt = 'Unsupported version control system: %(vcs)s'