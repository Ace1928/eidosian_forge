import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
@property
def _conn_msps(self):
    if not self._conn_msps_attr:
        try:
            namespace = self._MSPS_NAMESPACE % self._host
            self._conn_msps_attr = self._get_wmi_conn(namespace)
        except Exception:
            raise exceptions.OSWinException(_('Namespace %(namespace)s not found. Make sure FabricShieldedTools feature is installed.') % {'namespace': namespace})
    return self._conn_msps_attr