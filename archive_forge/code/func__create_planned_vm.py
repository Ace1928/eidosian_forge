import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def _create_planned_vm(self, conn_v2_local, conn_v2_remote, vm, ip_addr_list, dest_host):
    vsmsd = conn_v2_remote.Msvm_VirtualSystemMigrationSettingData(MigrationType=self._MIGRATION_TYPE_STAGED)[0]
    vsmsd.DestinationIPAddressList = ip_addr_list
    migration_setting_data = vsmsd.GetText_(1)
    LOG.debug('Creating planned VM for VM: %s', vm.ElementName)
    migr_svc = conn_v2_remote.Msvm_VirtualSystemMigrationService()[0]
    job_path, ret_val = migr_svc.MigrateVirtualSystemToHost(ComputerSystem=vm.path_(), DestinationHost=dest_host, MigrationSettingData=migration_setting_data)
    self._jobutils.check_ret_val(ret_val, job_path)
    return conn_v2_local.Msvm_PlannedComputerSystem(Name=vm.Name)[0]