class BzrRenameFailedError(BzrMoveFailedError):
    _fmt = 'Could not rename %(from_path)s%(operator)s %(to_path)s%(_has_extra)s%(extra)s'

    def __init__(self, from_path, to_path, extra=None):
        BzrMoveFailedError.__init__(self, from_path, to_path, extra)