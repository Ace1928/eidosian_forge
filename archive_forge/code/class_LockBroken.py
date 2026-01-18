class LockBroken(LockError):
    _fmt = 'Lock was broken while still open: %(lock)s - check storage consistency!'
    internal_error = False

    def __init__(self, lock):
        self.lock = lock