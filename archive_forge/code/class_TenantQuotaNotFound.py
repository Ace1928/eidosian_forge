from oslo_utils import excutils
from neutron_lib._i18n import _
class TenantQuotaNotFound(NotFound):
    message = _('Quota for tenant %(tenant_id)s could not be found.')