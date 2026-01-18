from __future__ import division
import datetime
import json
import logging
import os
import tempfile
from . import base
from ..discovery_cache import DISCOVERY_DOC_MAX_AGE
def _read_or_initialize_cache(f):
    f.file_handle().seek(0)
    try:
        cache = json.load(f.file_handle())
    except Exception:
        cache = {}
        f.file_handle().truncate(0)
        f.file_handle().seek(0)
        json.dump(cache, f.file_handle())
    return cache