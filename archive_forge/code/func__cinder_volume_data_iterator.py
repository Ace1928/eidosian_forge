import contextlib
import errno
import importlib
import logging
import math
import os
import shlex
import socket
import time
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import exceptions as keystone_exc
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from keystoneauth1 import token_endpoint as ksa_token_endpoint
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_utils import strutils
from oslo_utils import units
from glance_store._drivers.cinder import base
from glance_store import capabilities
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
import glance_store.location
from the service catalog, and current context's user and project are used.
def _cinder_volume_data_iterator(self, client, volume, max_size, offset=0, chunk_size=None, partial_length=None):
    chunk_size = chunk_size if chunk_size else self.READ_CHUNKSIZE
    partial = partial_length is not None
    with self._open_cinder_volume(client, volume, 'rb') as fp:
        if offset:
            fp.seek(offset)
            max_size -= offset
        while True:
            if partial:
                size = min(chunk_size, partial_length, max_size)
            else:
                size = min(chunk_size, max_size)
            chunk = fp.read(size)
            if chunk:
                yield chunk
                max_size -= len(chunk)
                if max_size <= 0:
                    break
                if partial:
                    partial_length -= len(chunk)
                    if partial_length <= 0:
                        break
            else:
                break