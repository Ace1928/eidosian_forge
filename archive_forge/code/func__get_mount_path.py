import hashlib
import logging
import os
import socket
from oslo_config import cfg
from glance_store._drivers.cinder import base
from glance_store.common import cinder_utils
from glance_store.common import fs_mount as mount
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _
def _get_mount_path(self, share, mount_point_base):
    """Returns the mount path prefix using the mount point base and share.

        :returns: The mount path prefix.
        """
    return os.path.join(self.mount_point_base, NfsBrickConnector.get_hash_str(share))