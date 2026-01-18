class HardLinkNotSupported(PathError):
    _fmt = 'Hard-linking "%(path)s" is not supported'