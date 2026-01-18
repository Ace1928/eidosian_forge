import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def _parse_mount_table(exit_code, output):
    """Parses the output of ``mount`` to produce (path, fs_type) pairs

    Separated from _generate_cifs_table to enable testing logic with real
    outputs
    """
    if exit_code != 0:
        return []
    pattern = re.compile('.*? on (/.*?) (?:type |\\()([^\\s,\\)]+)')
    matches = [(l, pattern.match(l)) for l in output.strip().splitlines() if l]
    mount_info = sorted((match.groups() for _, match in matches if match is not None), key=lambda x: len(x[0]), reverse=True)
    cifs_paths = [path for path, fstype in mount_info if fstype.lower() == 'cifs']
    for line, match in matches:
        if match is None:
            fmlogger.debug("Cannot parse mount line: '%s'", line)
    return [mount for mount in mount_info if any((mount[0].startswith(path) for path in cifs_paths))]