class LockCorrupt(LockError):
    _fmt = "Lock is apparently held, but corrupted: %(corruption_info)s\nUse 'brz break-lock' to clear it"
    internal_error = False

    def __init__(self, corruption_info, file_data=None):
        self.corruption_info = corruption_info
        self.file_data = file_data