class DistutilsFileError(DistutilsError):
    """Any problems in the filesystem: expected file not found, etc.
    Typically this is for problems that we detect before OSError
    could be raised."""
    pass