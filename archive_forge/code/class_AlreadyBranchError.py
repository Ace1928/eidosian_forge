class AlreadyBranchError(PathError):
    _fmt = 'Already a branch: "%(path)s".'