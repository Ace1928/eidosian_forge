class AlreadyCommitted(LockError):
    _fmt = 'A rollback was requested, but is not able to be accomplished.'

    def __init__(self):
        pass