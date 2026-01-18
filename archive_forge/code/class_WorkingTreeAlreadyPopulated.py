class WorkingTreeAlreadyPopulated(InternalBzrError):
    _fmt = 'Working tree already populated in "%(base)s"'

    def __init__(self, base):
        self.base = base