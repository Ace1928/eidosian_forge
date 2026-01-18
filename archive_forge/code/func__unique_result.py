from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
@staticmethod
def _unique_result(objects, resource_name):
    n = len(objects)
    if n == 0:
        raise exceptions.NotFound(resource=resource_name)
    elif n > 1:
        raise exceptions.OSWinException(_('Duplicate resource name found: %s') % resource_name)
    else:
        return objects[0]