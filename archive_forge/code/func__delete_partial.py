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
def _delete_partial(filepath, iid):
    try:
        os.unlink(filepath)
    except Exception as e:
        msg = _('Unable to remove partial image data for image %(iid)s: %(e)s')
        LOG.error(msg % dict(iid=iid, e=encodeutils.exception_to_unicode(e)))