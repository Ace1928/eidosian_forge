def S_IFMT(mode):
    """Return the portion of the file's mode that describes the
    file type.
    """
    return mode & 61440