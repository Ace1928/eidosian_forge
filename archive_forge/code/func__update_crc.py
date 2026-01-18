from __future__ import annotations
import csv
import hashlib
import os.path
import re
import stat
import time
from io import StringIO, TextIOWrapper
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from wheel.cli import WheelError
from wheel.util import log, urlsafe_b64decode, urlsafe_b64encode
def _update_crc(newdata):
    eof = ef._eof
    update_crc_orig(newdata)
    running_hash.update(newdata)
    if eof and running_hash.digest() != expected_hash:
        raise WheelError(f"Hash mismatch for file '{ef_name}'")