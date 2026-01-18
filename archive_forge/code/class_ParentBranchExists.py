class ParentBranchExists(AlreadyBranchError):
    _fmt = 'Parent branch already exists: "%(path)s".'