from fnmatch import fnmatch, fnmatchcase
def _match_path(pathname, included_patterns, excluded_patterns, case_sensitive=True):
    """Internal function same as :func:`match_path` but does not check arguments.

    Doctests::
        >>> _match_path("/users/gorakhargosh/foobar.py", ["*.py"], ["*.PY"], True)
        True
        >>> _match_path("/users/gorakhargosh/FOOBAR.PY", ["*.py"], ["*.PY"], True)
        False
        >>> _match_path("/users/gorakhargosh/foobar/", ["*.py"], ["*.txt"], False)
        False
        >>> _match_path("/users/gorakhargosh/FOOBAR.PY", ["*.py"], ["*.PY"], False)
        Traceback (most recent call last):
            ...
        ValueError: conflicting patterns `set(['*.py'])` included and excluded
    """
    if not case_sensitive:
        included_patterns = set(map(_string_lower, included_patterns))
        excluded_patterns = set(map(_string_lower, excluded_patterns))
    else:
        included_patterns = set(included_patterns)
        excluded_patterns = set(excluded_patterns)
    common_patterns = included_patterns & excluded_patterns
    if common_patterns:
        raise ValueError('conflicting patterns `%s` included and excluded' % common_patterns)
    return match_path_against(pathname, included_patterns, case_sensitive) and (not match_path_against(pathname, excluded_patterns, case_sensitive))