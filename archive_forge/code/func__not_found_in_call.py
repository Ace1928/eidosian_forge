from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def _not_found_in_call(self, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as ex:
        self.client_plugin().ignore_not_found(ex)
        return True
    else:
        return False