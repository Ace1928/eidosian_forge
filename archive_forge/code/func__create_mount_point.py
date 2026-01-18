import os
import re
import warnings
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.remotefs import remotefs
def _create_mount_point(self, share, use_local_path):
    mnt_point = self.get_mount_point(share)
    share_norm_path = self._get_share_norm_path(share)
    symlink_dest = share_norm_path if not use_local_path else self.get_local_share_path(share)
    if not os.path.isdir(self._mount_base):
        os.makedirs(self._mount_base)
    if os.path.exists(mnt_point):
        if not self._pathutils.is_symlink(mnt_point):
            raise exception.BrickException(_("Link path already exists and it's not a symlink"))
    else:
        self._pathutils.create_sym_link(mnt_point, symlink_dest)