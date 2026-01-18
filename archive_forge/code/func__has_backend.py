from __future__ import absolute_import
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.utils import to_bytes
from passlib.utils.compat import PYPY
def _has_backend(name):
    try:
        _set_backend(name, dryrun=True)
        return True
    except exc.MissingBackendError:
        return False