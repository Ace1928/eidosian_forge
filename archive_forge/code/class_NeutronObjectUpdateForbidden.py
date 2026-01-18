from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronObjectUpdateForbidden(exceptions.NeutronException):
    message = _('Unable to update the following object fields: %(fields)s')