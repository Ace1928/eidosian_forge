from oslo_utils import excutils
from neutron_lib._i18n import _
class TenantIdProjectIdFilterConflict(BadRequest):
    message = _('Both tenant_id and project_id passed as filters.')