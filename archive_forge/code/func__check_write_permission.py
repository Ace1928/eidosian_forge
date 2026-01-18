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
def _check_write_permission(self, datadir):
    """
        Checks if directory created to write image files has
        write permission.

        :datadir is a directory path in which glance wites image files.
        :raises: BadStoreConfiguration exception if datadir is read-only.
        """
    if not os.access(datadir, os.W_OK):
        msg = _('Permission to write in %s denied') % datadir
        LOG.exception(msg)
        raise exceptions.BadStoreConfiguration(store_name='filesystem', reason=msg)