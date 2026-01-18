import warnings
from ._common import files, as_file
def _get_encoding_arg(path_names, encoding):
    if encoding is _MISSING:
        if len(path_names) > 1:
            raise TypeError("'encoding' argument required with multiple path names")
        else:
            return 'utf-8'
    return encoding