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
def _create_image_directories(self, directory_paths):
    """
        Create directories to write image files if
        it does not exist.

        :directory_paths is a list of directories belonging to glance store.
        :raises: BadStoreConfiguration exception if creating a directory fails.
        """
    for datadir in directory_paths:
        if os.path.exists(datadir):
            self._check_write_permission(datadir)
            self._set_exec_permission(datadir)
        else:
            msg = _('Directory to write image files does not exist (%s). Creating.') % datadir
            LOG.info(msg)
            try:
                os.makedirs(datadir)
                self._check_write_permission(datadir)
                self._set_exec_permission(datadir)
            except (IOError, OSError):
                if os.path.exists(datadir):
                    self._check_write_permission(datadir)
                    self._set_exec_permission(datadir)
                    continue
                reason = _('Unable to create datadir: %s') % datadir
                LOG.error(reason)
                raise exceptions.BadStoreConfiguration(store_name='filesystem', reason=reason)