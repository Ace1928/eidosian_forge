from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class InvalidDestinationPort(qexception.NotFound):
    message = _('Destination Port %(port)s does not exist')