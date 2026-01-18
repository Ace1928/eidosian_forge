import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def _live_migrate_vm(self, conn_v2_local, vm, planned_vm, rmt_ip_addr_list, new_resource_setting_data, dest_host, migration_type):
    vsmsd = conn_v2_local.Msvm_VirtualSystemMigrationSettingData(MigrationType=migration_type)[0]
    vsmsd.DestinationIPAddressList = rmt_ip_addr_list
    if planned_vm:
        vsmsd.DestinationPlannedVirtualSystemId = planned_vm.Name
    migration_setting_data = vsmsd.GetText_(1)
    migr_svc = conn_v2_local.Msvm_VirtualSystemMigrationService()[0]
    LOG.debug('Starting live migration for VM: %s', vm.ElementName)
    job_path, ret_val = migr_svc.MigrateVirtualSystemToHost(ComputerSystem=vm.path_(), DestinationHost=dest_host, MigrationSettingData=migration_setting_data, NewResourceSettingData=new_resource_setting_data)
    self._jobutils.check_ret_val(ret_val, job_path)