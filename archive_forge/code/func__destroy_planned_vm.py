from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def _destroy_planned_vm(self, planned_vm):
    LOG.debug('Destroying existing planned VM: %s', planned_vm.ElementName)
    job_path, ret_val = self._vs_man_svc.DestroySystem(planned_vm.path_())
    self._jobutils.check_ret_val(ret_val, job_path)