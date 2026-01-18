import logging
import os
import tempfile
import shutil
import json
from subprocess import check_call, check_output
from tarfile import TarFile
from dateutil.zoneinfo import METADATA_FN, ZONEFILENAME
def _print_on_nosuchfile(e):
    """Print helpful troubleshooting message

    e is an exception raised by subprocess.check_call()

    """
    if e.errno == 2:
        logging.error("Could not find zic. Perhaps you need to install libc-bin or some other package that provides it, or it's not in your PATH?")