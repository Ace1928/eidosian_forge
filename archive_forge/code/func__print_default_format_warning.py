import contextlib
import ftplib
import gzip
import os
import re
import shutil
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.request import urlcleanup
@staticmethod
def _print_default_format_warning(file_format):
    """Print a warning to stdout (PRIVATE).

        Temporary warning (similar to a deprecation warning) that files
        are being downloaded in mmCIF.
        """
    if file_format is None:
        sys.stderr.write('WARNING: The default download format has changed from PDB to PDBx/mmCif\n')
        return 'mmCif'
    return file_format