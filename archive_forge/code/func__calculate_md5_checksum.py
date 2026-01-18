from __future__ import absolute_import
import argparse
import os
import io
import json
import logging
import sys
import errno
import hashlib
import math
import shutil
import tempfile
from functools import partial
def _calculate_md5_checksum(fname):
    """Calculate the checksum of the file, exactly same as md5-sum linux util.

    Parameters
    ----------
    fname : str
        Path to the file.

    Returns
    -------
    str
        MD5-hash of file names as `fname`.

    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()