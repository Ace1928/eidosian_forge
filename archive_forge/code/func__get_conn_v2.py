import platform
from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import migrationutils
from os_win.utils.compute import vmutils
def _get_conn_v2(self, host='localhost'):
    try:
        return self._get_wmi_obj(self._wmi_namespace % host, compatibility_mode=True)
    except exceptions.x_wmi as ex:
        LOG.exception('Get version 2 connection error')
        if ex.com_error.hresult == -2147217394:
            msg = _('Live migration is not supported on target host "%s"') % host
        elif ex.com_error.hresult == -2147023174:
            msg = _('Target live migration host "%s" is unreachable') % host
        else:
            msg = _('Live migration failed: %r') % ex
        raise exceptions.HyperVException(msg)