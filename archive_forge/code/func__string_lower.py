from fnmatch import fnmatch, fnmatchcase
def _string_lower(s):
    """
    Convenience function to lowercase a string (the :mod:`string` module is
    deprecated/removed in Python 3.0).

    :param s:
        The string which will be lowercased.
    :returns:
        Lowercased copy of string s.
    """
    return s.lower()