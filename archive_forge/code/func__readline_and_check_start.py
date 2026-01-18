import warnings
from Bio import BiopythonDeprecationWarning
def _readline_and_check_start(handle, start):
    """Read the first line and evaluate that begisn with the correct start (PRIVATE)."""
    line = handle.readline()
    if not line.startswith(start):
        raise ValueError(f'I expected {start!r} but got {line!r}')
    return line