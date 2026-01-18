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
def _call_online_extend(self, client, volume, size_gb):
    size_gb += 1
    LOG.debug('Extending (online) volume %(volume_id)s to %(size)s GB.', {'volume_id': volume.id, 'size': size_gb})
    self.volume_api.extend_volume(client, volume, size_gb)
    try:
        volume = self._wait_volume_status(volume, 'extending', 'in-use')
        size_gb = volume.size
        return size_gb
    except exceptions.BackendException:
        raise exceptions.StorageFull()