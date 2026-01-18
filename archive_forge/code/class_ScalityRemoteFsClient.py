import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
class ScalityRemoteFsClient(RemoteFsClient):

    def __init__(self, mount_type, root_helper, execute=None, *args, **kwargs):
        super(ScalityRemoteFsClient, self).__init__(mount_type, root_helper, *args, execute=execute, **kwargs)
        self._mount_type = mount_type
        self._mount_base = kwargs.get('scality_mount_point_base', '').rstrip('/')
        if not self._mount_base:
            raise exception.InvalidParameterValue(err=_('scality_mount_point_base required'))
        self._mount_options = None

    def get_mount_point(self, device_name):
        return os.path.join(self._mount_base, device_name, '00')

    def mount(self, share, flags=None):
        """Mount the Scality ScaleOut FS.

        The `share` argument is ignored because you can't mount several
        SOFS at the same type on a single server. But we want to keep the
        same method signature for class inheritance purpose.
        """
        if self._mount_base in self._read_mounts():
            LOG.debug('Already mounted: %s', self._mount_base)
            return
        self._execute('mkdir', '-p', self._mount_base, check_exit_code=0)
        super(ScalityRemoteFsClient, self)._do_mount('sofs', '/etc/sfused.conf', self._mount_base)