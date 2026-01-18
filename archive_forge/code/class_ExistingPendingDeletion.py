class ExistingPendingDeletion(BzrError):
    _fmt = 'This tree contains left-over files from a failed operation.\n    Please examine %(pending_deletion)s to see if it contains any files you\n    wish to keep, and delete it when you are done.'

    def __init__(self, pending_deletion):
        BzrError.__init__(self, pending_deletion=pending_deletion)