import os
import platform
import sys
import subprocess
import re
import warnings
from Bio import BiopythonDeprecationWarning
def _escape_filename(filename):
    """Escape filenames with spaces by adding quotes (PRIVATE).

    Note this will not add quotes if they are already included:

    >>> print((_escape_filename('example with spaces')))
    "example with spaces"
    >>> print((_escape_filename('"example with spaces"')))
    "example with spaces"
    >>> print((_escape_filename(1)))
    1

    Note the function is more generic than the name suggests, since it
    is used to add quotes around any string arguments containing spaces.
    """
    if not isinstance(filename, str):
        return filename
    if ' ' not in filename:
        return filename
    if filename.startswith('"') and filename.endswith('"'):
        return filename
    else:
        return f'"{filename}"'