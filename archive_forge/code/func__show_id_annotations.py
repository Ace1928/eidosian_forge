import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def _show_id_annotations(annotations, to_file, full, encoding):
    if not annotations:
        return
    last_rev_id = None
    max_origin_len = max((len(origin) for origin, text in annotations))
    for origin, text in annotations:
        if full or last_rev_id != origin:
            this = origin
        else:
            this = b''
        to_file.write('%*s | %s' % (max_origin_len, this.decode('utf-8'), text.decode(encoding)))
        last_rev_id = origin
    return