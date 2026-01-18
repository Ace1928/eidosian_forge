class LockActive(LockError):
    _fmt = "The lock for '%(lock_description)s' is in use and cannot be broken."
    internal_error = False

    def __init__(self, lock_description):
        self.lock_description = lock_description