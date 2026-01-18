import typing
import platform
import re
import six
def _has_drive_letter(path_snippet):
    """Check whether a path contains a drive letter.

    Arguments:
       path_snippet (str): a file path, relative or absolute.

    Example:
        >>> _has_drive_letter("D:/Data")
        True
        >>> _has_drive_letter(r"C:\\System32\\ test")
        True
        >>> _has_drive_letter("/tmp/abc:test")
        False

    """
    windows_drive_pattern = '.:[/\\\\].*$'
    return re.match(windows_drive_pattern, path_snippet) is not None