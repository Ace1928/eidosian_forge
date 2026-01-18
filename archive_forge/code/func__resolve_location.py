import errno
import logging
import os
import stat
import urllib
import jsonschema
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LW
import glance_store.location
from the filesystem store. The users running the services that are
@staticmethod
def _resolve_location(location):
    filepath = location.store_location.path
    if not os.path.exists(filepath):
        raise exceptions.NotFound(image=filepath)
    filesize = os.path.getsize(filepath)
    return (filepath, filesize)