class LockBreakMismatch(LockError):
    _fmt = 'Lock was released and re-acquired before being broken: %(lock)s: held by %(holder)r, wanted to break %(target)r'
    internal_error = False

    def __init__(self, lock, holder, target):
        self.lock = lock
        self.holder = holder
        self.target = target