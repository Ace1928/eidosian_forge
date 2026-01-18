from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def _validate_privileges(context, res_dict):
    if 'project_id' in res_dict and res_dict['project_id'] != context.project_id and (not (context.is_admin or context.is_advsvc)):
        msg = _("Specifying 'project_id' or 'tenant_id' other than the authenticated project in request requires admin or advsvc privileges")
        raise exc.HTTPBadRequest(msg)