from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class InvalidSourcePort(qexception.NotFound):
    message = _('Source Port  %(port)s does not exist')