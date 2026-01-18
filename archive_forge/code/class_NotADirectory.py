class NotADirectory(PathError):
    _fmt = '"%(path)s" is not a directory %(extra)s'