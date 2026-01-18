def S_ISBLK(mode):
    """Return True if mode is from a block special device file."""
    return S_IFMT(mode) == S_IFBLK