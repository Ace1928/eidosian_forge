from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def _calculate_data_hashes(data):
    _md5 = md5(usedforsecurity=False)
    _sha256 = hashlib.sha256()
    if hasattr(data, 'read'):
        for chunk in iter(lambda: data.read(8192), b''):
            _md5.update(chunk)
            _sha256.update(chunk)
    else:
        _md5.update(data)
        _sha256.update(data)
    return (_md5.hexdigest(), _sha256.hexdigest())