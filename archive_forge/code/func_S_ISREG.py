def S_ISREG(mode):
    """Return True if mode is from a regular file."""
    return S_IFMT(mode) == S_IFREG