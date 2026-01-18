import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _get_port_security_acls(self, port):
    """Returns a mutable list of Security Group Rule objects.

        Returns the list of Security Group Rule objects from the cache,
        otherwise it fetches and caches from the port's associated class.
        """
    if port.ElementName in self._sg_acl_sds:
        return self._sg_acl_sds[port.ElementName]
    acls = _wqlutils.get_element_associated_class(self._conn, self._PORT_EXT_ACL_SET_DATA, element_instance_id=port.InstanceID)
    if self._enable_cache:
        self._sg_acl_sds[port.ElementName] = acls
    return acls