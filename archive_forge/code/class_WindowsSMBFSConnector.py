import os
from os_win import utilsfactory
from os_brick.initiator.windows import base as win_conn_base
from os_brick.remotefs import windows_remotefs as remotefs
from os_brick import utils
class WindowsSMBFSConnector(win_conn_base.BaseWindowsConnector):

    def __init__(self, *args, **kwargs):
        super(WindowsSMBFSConnector, self).__init__(*args, **kwargs)
        self._local_path_for_loopback = kwargs.get('local_path_for_loopback', True)
        self._expect_raw_disk = kwargs.get('expect_raw_disk', False)
        self._remotefsclient = remotefs.WindowsRemoteFsClient(*args, mount_type='smbfs', **kwargs)
        self._smbutils = utilsfactory.get_smbutils()
        self._vhdutils = utilsfactory.get_vhdutils()
        self._diskutils = utilsfactory.get_diskutils()

    @staticmethod
    def get_connector_properties(*args, **kwargs):
        return {}

    @utils.trace
    def connect_volume(self, connection_properties):
        self.ensure_share_mounted(connection_properties)
        disk_path = self._get_disk_path(connection_properties)
        if self._expect_raw_disk:
            read_only = connection_properties.get('access_mode') == 'ro'
            self._vhdutils.attach_virtual_disk(disk_path, read_only=read_only)
            raw_disk_path = self._vhdutils.get_virtual_disk_physical_path(disk_path)
            dev_num = self._diskutils.get_device_number_from_device_name(raw_disk_path)
            self._diskutils.set_disk_offline(dev_num)
        else:
            raw_disk_path = None
        device_info = {'type': 'file', 'path': raw_disk_path if self._expect_raw_disk else disk_path}
        return device_info

    @utils.trace
    def disconnect_volume(self, connection_properties, device_info=None, force=False, ignore_errors=False):
        export_path = self._get_export_path(connection_properties)
        disk_path = self._get_disk_path(connection_properties)
        self._vhdutils.detach_virtual_disk(disk_path)
        self._remotefsclient.unmount(export_path)

    def _get_export_path(self, connection_properties):
        return connection_properties['export'].replace('/', '\\')

    def _get_disk_path(self, connection_properties):
        export_path = self._get_export_path(connection_properties)
        mount_base = self._remotefsclient.get_mount_base()
        use_local_path = self._local_path_for_loopback and self._smbutils.is_local_share(export_path)
        disk_dir = export_path
        if mount_base:
            disk_dir = self._remotefsclient.get_mount_point(export_path)
        elif use_local_path:
            disk_dir = self._remotefsclient.get_local_share_path(export_path)
        disk_name = connection_properties['name']
        disk_path = os.path.join(disk_dir, disk_name)
        return disk_path

    def get_search_path(self):
        return self._remotefsclient.get_mount_base()

    @utils.trace
    def get_volume_paths(self, connection_properties):
        return [self._get_disk_path(connection_properties)]

    def ensure_share_mounted(self, connection_properties):
        export_path = self._get_export_path(connection_properties)
        mount_options = connection_properties.get('options')
        self._remotefsclient.mount(export_path, mount_options)

    def extend_volume(self, connection_properties):
        raise NotImplementedError