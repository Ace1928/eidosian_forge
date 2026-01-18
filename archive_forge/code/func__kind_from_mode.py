import stat
def _kind_from_mode(stat_mode, _formats=_formats, _unknown='unknown'):
    """Generate a file kind from a stat mode. This is used in walkdirs.

    It's performance is critical: Do not mutate without careful benchmarking.
    """
    try:
        return _formats[stat_mode & 61440]
    except KeyError:
        return _unknown