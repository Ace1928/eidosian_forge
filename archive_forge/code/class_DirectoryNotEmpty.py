class DirectoryNotEmpty(PathError):
    _fmt = 'Directory not empty: "%(path)s"%(extra)s'