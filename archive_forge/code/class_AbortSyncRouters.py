from neutron_lib._i18n import _
from neutron_lib import exceptions
class AbortSyncRouters(exceptions.NeutronException):
    message = _('Aborting periodic_sync_routers_task due to an error.')