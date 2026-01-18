def S_ISLNK(mode):
    """Return True if mode is from a symbolic link."""
    return S_IFMT(mode) == S_IFLNK