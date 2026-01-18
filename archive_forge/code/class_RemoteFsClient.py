import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
class RemoteFsClient(executor.Executor):

    def __init__(self, mount_type, root_helper, execute=None, *args, **kwargs):
        super(RemoteFsClient, self).__init__(root_helper, *args, execute=execute, **kwargs)
        mount_type_to_option_prefix = {'nfs': 'nfs', 'cifs': 'smbfs', 'glusterfs': 'glusterfs', 'vzstorage': 'vzstorage', 'quobyte': 'quobyte', 'scality': 'scality'}
        if mount_type not in mount_type_to_option_prefix:
            raise exception.ProtocolNotSupported(protocol=mount_type)
        self._mount_type = mount_type
        option_prefix = mount_type_to_option_prefix[mount_type]
        self._mount_base: str
        self._mount_base = kwargs.get(option_prefix + '_mount_point_base')
        if not self._mount_base:
            raise exception.InvalidParameterValue(err=_('%s_mount_point_base required') % option_prefix)
        self._mount_options = kwargs.get(option_prefix + '_mount_options')
        if mount_type == 'nfs':
            self._check_nfs_options()

    def get_mount_base(self):
        return self._mount_base

    def _get_hash_str(self, base_str):
        """Return a string that represents hash of base_str (hex format)."""
        if isinstance(base_str, str):
            base_str = base_str.encode('utf-8')
        return md5(base_str, usedforsecurity=False).hexdigest()

    def get_mount_point(self, device_name: str):
        """Get Mount Point.

        :param device_name: example 172.18.194.100:/var/nfs
        """
        return os.path.join(self._mount_base, self._get_hash_str(device_name))

    def _read_mounts(self):
        """Returns a dict of mounts and their mountpoint

        Format reference:
        http://man7.org/linux/man-pages/man5/fstab.5.html
        """
        with open('/proc/mounts', 'r') as mounts:
            lines = [line.split() for line in mounts.read().splitlines() if line.strip()]
            return {line[1]: line[0] for line in lines if line[0] != '#'}

    def mount(self, share, flags=None):
        """Mount given share."""
        mount_path = self.get_mount_point(share)
        if mount_path in self._read_mounts():
            LOG.debug('Already mounted: %s', mount_path)
            return
        self._execute('mkdir', '-p', mount_path, check_exit_code=0)
        if self._mount_type == 'nfs':
            self._mount_nfs(share, mount_path, flags)
        else:
            self._do_mount(self._mount_type, share, mount_path, self._mount_options, flags)

    def _do_mount(self, mount_type, share, mount_path, mount_options=None, flags=None):
        """Mounts share based on the specified params."""
        mnt_cmd = ['mount', '-t', mount_type]
        if mount_options is not None:
            mnt_cmd.extend(['-o', mount_options])
        if flags is not None:
            mnt_cmd.extend(flags)
        mnt_cmd.extend([share, mount_path])
        try:
            self._execute(*mnt_cmd, root_helper=self._root_helper, run_as_root=True, check_exit_code=0)
        except processutils.ProcessExecutionError as exc:
            if 'already mounted' in exc.stderr:
                LOG.debug('Already mounted: %s', share)
                if share in self._read_mounts():
                    return
            LOG.error('Failed to mount %(share)s, reason: %(reason)s', {'share': share, 'reason': exc.stderr})
            raise

    def _mount_nfs(self, nfs_share, mount_path, flags=None):
        """Mount nfs share using present mount types."""
        mnt_errors = {}
        for mnt_type in sorted(self._nfs_mount_type_opts.keys(), reverse=True):
            options = self._nfs_mount_type_opts[mnt_type]
            try:
                self._do_mount('nfs', nfs_share, mount_path, options, flags)
                LOG.debug('Mounted %(sh)s using %(mnt_type)s.', {'sh': nfs_share, 'mnt_type': mnt_type})
                return
            except Exception as e:
                mnt_errors[mnt_type] = str(e)
                LOG.debug('Failed to do %s mount.', mnt_type)
        raise exception.BrickException(_('NFS mount failed for share %(sh)s. Error - %(error)s') % {'sh': nfs_share, 'error': mnt_errors})

    def _check_nfs_options(self):
        """Checks and prepares nfs mount type options."""
        self._nfs_mount_type_opts = {'nfs': self._mount_options}
        nfs_vers_opt_patterns = ['^nfsvers', '^vers', '^v[\\d]']
        for opt in nfs_vers_opt_patterns:
            if self._option_exists(self._mount_options, opt):
                return
        pnfs_opts = self._update_option(self._mount_options, 'vers', '4')
        pnfs_opts = self._update_option(pnfs_opts, 'minorversion', '1')
        self._nfs_mount_type_opts['pnfs'] = pnfs_opts

    def _option_exists(self, options, opt_pattern):
        """Checks if the option exists in nfs options and returns position."""
        options = [x.strip() for x in options.split(',')] if options else []
        pos = 0
        for opt in options:
            pos = pos + 1
            if re.match(opt_pattern, opt, flags=0):
                return pos
        return 0

    def _update_option(self, options, option, value=None):
        """Update option if exists else adds it and returns new options."""
        opts = [x.strip() for x in options.split(',')] if options else []
        pos = self._option_exists(options, option)
        if pos:
            opts.pop(pos - 1)
        opt = '%s=%s' % (option, value) if value else option
        opts.append(opt)
        return ','.join(opts) if len(opts) > 1 else opts[0]