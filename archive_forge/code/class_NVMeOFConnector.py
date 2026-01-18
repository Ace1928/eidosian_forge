from __future__ import annotations
import errno
import functools
import glob
import json
import os.path
import time
from typing import (Callable, Optional, Sequence, Type, Union)  # noqa: H301
import uuid as uuid_lib
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
class NVMeOFConnector(base.BaseLinuxConnector):
    """Connector class to attach/detach NVMe-oF volumes."""
    native_multipath_supported = None
    TIME_TO_CONNECT = 10

    def __init__(self, root_helper: str, driver: Optional[base.host_driver.HostDriver]=None, use_multipath: bool=False, device_scan_attempts: int=DEVICE_SCAN_ATTEMPTS_DEFAULT, *args, **kwargs) -> None:
        super(NVMeOFConnector, self).__init__(root_helper, driver, *args, device_scan_attemps=device_scan_attempts, **kwargs)
        self.use_multipath = use_multipath
        self._set_native_multipath_supported()
        if self.use_multipath and (not self.native_multipath_supported):
            LOG.warning('native multipath is not enabled')

    @staticmethod
    def get_search_path() -> str:
        """Necessary implementation for an os-brick connector."""
        return DEV_SEARCH_PATH

    def get_volume_paths(self, connection_properties: NVMeOFConnProps, device_info: Optional[dict[str, str]]=None) -> list[str]:
        """Return paths where the volume is present."""
        if device_info and device_info.get('path'):
            return [device_info['path']]
        device_path = connection_properties.device_path
        if device_path:
            return [device_path]
        LOG.warning('We are being called without the path information!')
        if connection_properties.is_replicated:
            if connection_properties.alias is None:
                raise exception.BrickException('Alias missing in connection info')
            return [RAID_PATH + connection_properties.alias]
        devices = connection_properties.get_devices()
        if connection_properties.is_replicated is None:
            if any((self._is_raid_device(dev) for dev in devices)):
                if connection_properties.alias is None:
                    raise exception.BrickException('Alias missing in connection info')
                return [RAID_PATH + connection_properties.alias]
        return devices

    @classmethod
    def nvme_present(cls: type) -> bool:
        """Check if the nvme CLI is present."""
        try:
            priv_rootwrap.custom_execute('nvme', 'version')
            return True
        except Exception as exc:
            if isinstance(exc, OSError) and exc.errno == errno.ENOENT:
                LOG.debug('nvme not present on system')
            else:
                LOG.warning('Unknown error when checking presence of nvme: %s', exc)
        return False

    @classmethod
    def get_connector_properties(cls, root_helper, *args, **kwargs) -> dict:
        """The NVMe-oF connector properties (initiator uuid and nqn.)"""
        execute = kwargs.get('execute') or priv_rootwrap.execute
        nvmf = NVMeOFConnector(root_helper=root_helper, execute=execute)
        ret = {}
        nqn = None
        hostid = None
        uuid = nvmf._get_host_uuid()
        suuid = priv_nvmeof.get_system_uuid()
        if cls.nvme_present():
            nqn = utils.get_host_nqn(suuid)
            hostid = utils.get_nvme_host_id(suuid)
        if hostid:
            ret['nvme_hostid'] = hostid
        if uuid:
            ret['uuid'] = uuid
        if suuid:
            ret['system uuid'] = suuid
        if nqn:
            ret['nqn'] = nqn
        ret['nvme_native_multipath'] = cls._set_native_multipath_supported()
        return ret

    def _get_host_uuid(self) -> Optional[str]:
        """Get the UUID of the first mounted filesystem."""
        cmd = ('findmnt', '-v', '/', '-n', '-o', 'SOURCE')
        try:
            lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
            blkid_cmd = ('blkid', lines.split('\n')[0], '-s', 'UUID', '-o', 'value')
            lines, _err = self._execute(*blkid_cmd, run_as_root=True, root_helper=self._root_helper)
            return lines.split('\n')[0]
        except putils.ProcessExecutionError as e:
            LOG.warning('Process execution error in _get_host_uuid: %s', e)
            return None

    @classmethod
    def _set_native_multipath_supported(cls):
        if cls.native_multipath_supported is None:
            cls.native_multipath_supported = cls._is_native_multipath_supported()
        return cls.native_multipath_supported

    @staticmethod
    def _is_native_multipath_supported():
        try:
            with open('/sys/module/nvme_core/parameters/multipath', 'rt') as f:
                return f.read().strip() == 'Y'
        except Exception:
            LOG.warning('Could not find nvme_core/parameters/multipath')
        return False

    @utils.trace
    @utils.connect_volume_prepare_result
    @base.synchronized('connect_volume', external=True)
    @NVMeOFConnProps.from_dictionary_parameter
    def connect_volume(self, connection_properties: NVMeOFConnProps) -> dict[str, str]:
        """Attach and discover the volume."""
        try:
            if connection_properties.is_replicated is False:
                LOG.debug('Starting non replicated connection')
                path = self._connect_target(connection_properties.targets[0])
            else:
                LOG.debug('Starting replicated connection')
                path = self._connect_volume_replicated(connection_properties)
        except Exception:
            self._try_disconnect_all(connection_properties)
            raise
        return {'type': 'block', 'path': path}

    def _do_multipath(self):
        return self.use_multipath and self.native_multipath_supported

    @utils.retry(exception.VolumeDeviceNotFound, interval=2)
    def _connect_target(self, target: Target) -> str:
        """Attach a specific target to present a volume on the system

        If we are already connected to any of the portals (and it's live) we
        send a rescan (because the backend may not support AER messages),
        otherwise we iterate through the portals trying to do an nvme-of
        connection.

        This method assumes that the controllers for the portals have already
        been set.  For example using the from_dictionary_parameter decorator
        in the NVMeOFConnProps class.

        Returns the path of the connected device.
        """
        connected = False
        missing_portals = []
        reconnecting_portals = []
        for portal in target.portals:
            state = portal.state
            if state == portal.LIVE:
                connected = True
                self.rescan(portal.controller)
            elif state == portal.MISSING:
                missing_portals.append(portal)
            elif state == portal.CONNECTING:
                LOG.debug('%s is reconnecting', portal)
                reconnecting_portals.append(portal)
            else:
                LOG.debug('%s exists but is %s', portal, state)
        do_multipath = self._do_multipath()
        if do_multipath or not connected:
            for portal in missing_portals:
                cmd = ['connect', '-a', portal.address, '-s', portal.port, '-t', portal.transport, '-n', target.nqn, '-Q', '128', '-l', '-1']
                if target.host_nqn:
                    cmd.extend(['-q', target.host_nqn])
                try:
                    self.run_nvme_cli(cmd)
                    connected = True
                except putils.ProcessExecutionError as exc:
                    if not (exc.exit_code in (70, errno.EALREADY) or (exc.exit_code == 1 and 'already connected' in exc.stderr + exc.stdout)):
                        LOG.error('Could not connect to %s: exit_code: %s, stdout: "%s", stderr: "%s",', portal, exc.exit_code, exc.stdout, exc.stderr)
                        continue
                    LOG.warning('Race condition with some other application when connecting to %s, please check your system configuration.', portal)
                    state = portal.state
                    if state == portal.LIVE:
                        connected = True
                    elif state == portal.CONNECTING:
                        reconnecting_portals.append(portal)
                    else:
                        LOG.error('Ignoring %s due to unknown state (%s)', portal, state)
                if not do_multipath:
                    break
        if not connected and reconnecting_portals:
            delay = self.TIME_TO_CONNECT + max((p.reconnect_delay for p in reconnecting_portals))
            LOG.debug('Waiting %s seconds for some nvme controllers to reconnect', delay)
            timeout = time.time() + delay
            while time.time() < timeout:
                time.sleep(1)
                if any((p.is_live for p in reconnecting_portals)):
                    LOG.debug('Reconnected')
                    connected = True
                    break
            LOG.debug('No controller reconnected')
        if not connected:
            raise exception.VolumeDeviceNotFound(device=target.nqn)
        target.set_portals_controllers()
        dev_path = target.find_device()
        return dev_path

    @utils.trace
    def _connect_volume_replicated(self, connection_properties: NVMeOFConnProps) -> str:
        """Connect to a replicated volume and prepare the RAID

        Connection properties must contain all the necessary replica
        information, even if there is only 1 replica.

        Returns the /dev/md path

        Raises VolumeDeviceNotFound when cannot present the device in the
        system.
        """
        host_device_paths = []
        if not connection_properties.alias:
            raise exception.BrickException('Alias missing in connection info')
        for replica in connection_properties.targets:
            try:
                rep_host_device_path = self._connect_target(replica)
                host_device_paths.append(rep_host_device_path)
            except Exception as ex:
                LOG.error('_connect_target: %s', ex)
        if not host_device_paths:
            raise exception.VolumeDeviceNotFound(device=connection_properties.targets)
        if connection_properties.is_replicated:
            device_path = self._handle_replicated_volume(host_device_paths, connection_properties)
        else:
            device_path = self._handle_single_replica(host_device_paths, connection_properties.alias)
        if nvmeof_agent:
            nvmeof_agent.NVMeOFAgent.ensure_running(self)
        return device_path

    def _handle_replicated_volume(self, host_device_paths: list[str], conn_props: NVMeOFConnProps) -> str:
        """Assemble the raid from found devices."""
        path_in_raid = False
        for dev_path in host_device_paths:
            path_in_raid = self._is_device_in_raid(dev_path)
            if path_in_raid:
                break
        device_path = RAID_PATH + conn_props.alias
        if path_in_raid:
            self.stop_and_assemble_raid(host_device_paths, device_path, False)
        else:
            paths_found = len(host_device_paths)
            if conn_props.replica_count > paths_found:
                LOG.error('Cannot create MD as %s out of %s legs were found.', paths_found, conn_props.replica_count)
                raise exception.VolumeDeviceNotFound(device=conn_props.alias)
            self.create_raid(host_device_paths, '1', conn_props.alias, conn_props.alias, False)
        return device_path

    def _handle_single_replica(self, host_device_paths: list[str], volume_alias: str) -> str:
        """Assemble the raid from a single device."""
        if self._is_raid_device(host_device_paths[0]):
            md_path = RAID_PATH + volume_alias
            self.stop_and_assemble_raid(host_device_paths, md_path, False)
            return md_path
        return host_device_paths[0]

    @utils.trace
    @base.synchronized('connect_volume', external=True)
    @utils.connect_volume_undo_prepare_result(unlink_after=True)
    def disconnect_volume(self, connection_properties: dict, device_info: dict[str, str], force: bool=False, ignore_errors: bool=False) -> None:
        """Flush the volume.

        Disconnect of volumes happens on storage system side. Here we could
        remove connections to subsystems if no volumes are left. But new
        volumes can pop up asynchronously in the meantime. So the only thing
        left is flushing or disassembly of a correspondng RAID device.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes as
                                      described in connect_volume but also with
                                      the "device_path" key containing the path
                                      to the volume that was connected (this is
                                      added by Nova).
        :type connection_properties: dict

        :param device_info: historical difference, but same as connection_props
        :type device_info: dict
        """
        conn_props = NVMeOFConnProps(connection_properties)
        try:
            device_path = self.get_volume_paths(conn_props, device_info)[0]
        except IndexError:
            LOG.warning("Cannot find the device for %s, assuming it's not there.", conn_props.cinder_volume_id or conn_props.targets[0].nqn)
            return
        exc = exception.ExceptionChainer()
        if not os.path.exists(device_path):
            LOG.warning('Trying to disconnect device %(device_path)s, but it is not connected. Skipping.', {'device_path': device_path})
            return
        if device_path.startswith(RAID_PATH):
            with exc.context(force, 'Failed to end raid %s', device_path):
                self.end_raid(device_path)
        else:
            with exc.context(force, 'Failed to flush %s', device_path):
                self._linuxscsi.flush_device_io(device_path)
        self._try_disconnect_all(conn_props, exc)
        if exc:
            LOG.warning('There were errors removing %s', device_path)
            if not ignore_errors:
                raise exc

    def _try_disconnect_all(self, conn_props: NVMeOFConnProps, exc: Optional[exception.ExceptionChainer]=None) -> None:
        """Disconnect all subsystems that are not being used.

        Only sees if it has to disconnect this connection properties portals,
        leaves other alone.

        Since this is unrelated to the flushing of the devices failures will be
        logged but they won't be raised.
        """
        if exc is None:
            exc = exception.ExceptionChainer()
        for target in conn_props.targets:
            target.set_portals_controllers()
            for portal in target.portals:
                with exc.context(True, 'Failed to disconnect %s', portal):
                    self._try_disconnect(portal)

    def _try_disconnect(self, portal: Portal) -> None:
        """Disconnect a specific subsystem if it's safe.

        Only disconnect if it has no namespaces left or has only one left and
        it is from this connection.
        """
        LOG.debug('Checking if %s can be disconnected', portal)
        if portal.can_disconnect():
            self._execute('nvme', 'disconnect', '-d', '/dev/' + portal.controller, root_helper=self._root_helper, run_as_root=True)

    @staticmethod
    def _get_sizes_from_lba(ns_data: dict) -> tuple[Optional[int], Optional[int]]:
        """Return size in bytes and the nsze of the volume from NVMe NS data.

        nsze is the namespace size that defines the total size of the namespace
        in logical blocks (LBA 0 through n-1), as per NVMe-oF specs.

        Returns a tuple of nsze and size
        """
        try:
            lbads = ns_data['lbafs'][0]['ds']
            if len(ns_data['lbafs']) != 1 or lbads < 9:
                LOG.warning('Cannot calculate new size with LBAs')
                return (None, None)
            nsze = ns_data['nsze']
            new_size = nsze * (1 << lbads)
        except Exception:
            return (None, None)
        LOG.debug('New volume size is %s and nsze is %s', new_size, nsze)
        return (nsze, new_size)

    @utils.trace
    @base.synchronized('extend_volume', external=True)
    @utils.connect_volume_undo_prepare_result
    def extend_volume(self, connection_properties: dict[str, str]) -> int:
        """Update an attached volume to reflect the current size after extend

        The only way to reflect the new size of an NVMe-oF volume on the host
        is a rescan, which rescans the whole subsystem.  This is a problem on
        attach_volume and detach_volume, but not here, since we will have at
        least the namespace we are operating on in the subsystem.

        The tricky part is knowing when a rescan has already been completed and
        the volume size on sysfs is final.  The rescan may already have
        happened before this method is called due to an AER message or we may
        need to trigger it here.

        Scans can be triggered manually with 'nvme ns-rescan' or writing 1 in
        configf's rescan file, or they can be triggered indirectly when calling
        the 'nvme list', 'nvme id-ns', or even using the 'nvme admin-passthru'
        command.

        Even after getting the new size with any of the NVMe commands above, we
        still need to wait until this is reflected on the host device, because
        we cannot return to the caller until the new size is in effect.

        If we don't see the new size taking effect on the system after 5
        seconds, or if we cannot get the new size with nvme, then we rescan in
        the latter and in both cases we blindly wait 5 seconds and return
        whatever size is present.

        For replicated volumes, the RAID needs to be extended.
        """
        conn_props = NVMeOFConnProps(connection_properties)
        try:
            device_path = self.get_volume_paths(conn_props)[0]
        except IndexError:
            raise exception.VolumeDeviceNotFound()
        if device_path.startswith(RAID_PATH):
            self.run_mdadm(('mdadm', '--grow', '--size', 'max', device_path))
        else:
            dev_name = os.path.basename(device_path)
            ctrl_name = dev_name.rsplit('n', 1)[0]
            nsze: Optional[Union[str, int]] = None
            try:
                out, err = self._execute('nvme', 'id-ns', '-ojson', device_path, run_as_root=True, root_helper=self._root_helper)
                ns_data = json.loads(out)
                nsze, new_size = self._get_sizes_from_lba(ns_data)
            except Exception as exc:
                LOG.warning('Failed to get id-ns %s', exc)
                self.rescan(ctrl_name)
            if nsze:
                nsze = str(nsze)
                for x in range(10):
                    current_nsze = blk_property('size', dev_name)
                    if current_nsze == nsze:
                        return new_size
                    LOG.debug('Sysfs size is still %s', current_nsze)
                    time.sleep(0.5)
                LOG.warning('Timeout waiting for sysfs to reflect the right volume size.')
            LOG.info('Wait 5 seconds and return whatever size is present')
            time.sleep(5)
        size = utils.get_device_size(self, device_path)
        if size is None:
            raise exception.BrickException('get_device_size returned non-numeric size')
        return size

    def run_mdadm(self, cmd: Sequence[str], raise_exception: bool=False) -> Optional[str]:
        cmd_output = None
        try:
            lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
            for line in lines.split('\n'):
                cmd_output = line
                break
        except putils.ProcessExecutionError as ex:
            LOG.warning('[!] Could not run mdadm: %s', str(ex))
            if raise_exception:
                raise ex
        return cmd_output

    def _is_device_in_raid(self, device_path: str) -> bool:
        cmd = ['mdadm', '--examine', device_path]
        raid_expected = device_path + ':'
        try:
            lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
            for line in lines.split('\n'):
                if line == raid_expected:
                    return True
                else:
                    return False
        except putils.ProcessExecutionError:
            pass
        return False

    @staticmethod
    def ks_readlink(dest: str) -> str:
        try:
            return os.readlink(dest)
        except Exception:
            return ''

    @staticmethod
    def get_md_name(device_name: str) -> Optional[str]:
        try:
            with open('/proc/mdstat', 'r') as f:
                lines = [line.split(' ')[0] for line in f if device_name in line]
                if lines:
                    return lines[0]
        except Exception as exc:
            LOG.debug('[!] Could not find md name for %s in mdstat: %s', device_name, exc)
        return None

    def stop_and_assemble_raid(self, drives: list[str], md_path: str, read_only: bool) -> None:
        md_name = None
        i = 0
        assembled = False
        link = ''
        while i < 5 and (not assembled):
            for drive in drives:
                device_name = drive[5:]
                md_name = self.get_md_name(device_name)
                link = NVMeOFConnector.ks_readlink(md_path)
                if link != '':
                    link = os.path.basename(link)
                if md_name and md_name == link:
                    return
                LOG.debug('sleeping 1 sec -allow auto assemble link = %(link)s md path = %(md_path)s', {'link': link, 'md_path': md_path})
                time.sleep(1)
            if md_name and md_name != link:
                self.stop_raid(md_name)
            try:
                assembled = self.assemble_raid(drives, md_path, read_only)
            except Exception:
                i += 1

    def assemble_raid(self, drives: list[str], md_path: str, read_only: bool) -> bool:
        cmd = ['mdadm', '--assemble', '--run', md_path]
        if read_only:
            cmd.append('-o')
        for i in range(len(drives)):
            cmd.append(drives[i])
        try:
            self.run_mdadm(cmd, True)
        except putils.ProcessExecutionError as ex:
            LOG.warning('[!] Could not _assemble_raid: %s', str(ex))
            raise ex
        return True

    def create_raid(self, drives: list[str], raid_type: str, device_name: str, name: str, read_only: bool) -> None:
        cmd = ['mdadm']
        num_drives = len(drives)
        cmd.append('-C')
        if read_only:
            cmd.append('-o')
        cmd.append(device_name)
        cmd.append('-R')
        if name:
            cmd.append('-N')
            cmd.append(name)
        cmd.append('--level')
        cmd.append(raid_type)
        cmd.append('--raid-devices=' + str(num_drives))
        cmd.append('--bitmap=internal')
        cmd.append('--homehost=any')
        cmd.append('--failfast')
        cmd.append('--assume-clean')
        for i in range(len(drives)):
            cmd.append(drives[i])
        LOG.debug('[!] cmd = %s', cmd)
        self.run_mdadm(cmd)
        for i in range(60):
            try:
                is_exist = os.path.exists(RAID_PATH + name)
                LOG.debug('[!] md is_exist = %s', is_exist)
                if is_exist:
                    return
                time.sleep(1)
            except Exception:
                LOG.debug('[!] Exception_wait_raid!')
        msg = _('md: /dev/md/%s not found.') % name
        LOG.error(msg)
        raise exception.NotFound(message=msg)

    def end_raid(self, device_path: str) -> None:
        raid_exists = self.is_raid_exists(device_path)
        if raid_exists:
            for i in range(10):
                try:
                    cmd_out = self.stop_raid(device_path, True)
                    if not cmd_out:
                        break
                except Exception:
                    time.sleep(1)
            try:
                is_exist = os.path.exists(device_path)
                LOG.debug('[!] is_exist = %s', is_exist)
                if is_exist:
                    self.remove_raid(device_path)
                    os.remove(device_path)
            except Exception:
                LOG.debug('[!] Exception_stop_raid!')

    def stop_raid(self, md_path: str, raise_exception: bool=False) -> Optional[str]:
        cmd = ['mdadm', '--stop', md_path]
        LOG.debug('[!] cmd = %s', cmd)
        cmd_out = self.run_mdadm(cmd, raise_exception)
        return cmd_out

    def is_raid_exists(self, device_path: str) -> bool:
        cmd = ['mdadm', '--detail', device_path]
        LOG.debug('[!] cmd = %s', cmd)
        raid_expected = device_path + ':'
        try:
            lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper)
            for line in lines.split('\n'):
                LOG.debug('[!] line = %s', line)
                if line == raid_expected:
                    return True
                else:
                    return False
        except putils.ProcessExecutionError:
            pass
        return False

    def remove_raid(self, device_path: str) -> None:
        cmd = ['mdadm', '--remove', device_path]
        LOG.debug('[!] cmd = %s', cmd)
        self.run_mdadm(cmd)

    def _is_raid_device(self, device: str) -> bool:
        return self._get_fs_type(device) == 'linux_raid_member'

    def _get_fs_type(self, device_path: str) -> Optional[str]:
        cmd = ['blkid', device_path, '-s', 'TYPE', '-o', 'value']
        LOG.debug('[!] cmd = %s', cmd)
        fs_type = None
        lines, err = self._execute(*cmd, run_as_root=True, root_helper=self._root_helper, check_exit_code=False)
        fs_type = lines.split('\n')[0]
        return fs_type or None

    def run_nvme_cli(self, nvme_command: Sequence[str], **kwargs) -> tuple[str, str]:
        """Run an nvme cli command and return stdout and stderr output."""
        out, err = self._execute('nvme', *nvme_command, run_as_root=True, root_helper=self._root_helper, check_exit_code=True)
        msg = 'nvme %(nvme_command)s: stdout=%(out)s stderr=%(err)s' % {'nvme_command': nvme_command, 'out': out, 'err': err}
        LOG.debug('[!] %s', msg)
        return (out, err)

    def rescan(self, controller_name: str) -> None:
        """Rescan an nvme controller."""
        nvme_command = ('ns-rescan', DEV_SEARCH_PATH + controller_name)
        try:
            self.run_nvme_cli(nvme_command)
        except Exception as e:
            raise exception.CommandExecutionFailed(e, cmd=nvme_command)