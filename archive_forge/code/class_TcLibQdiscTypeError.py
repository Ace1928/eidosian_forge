from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class TcLibQdiscTypeError(e.NeutronException):
    message = _('TC Qdisc type %(qdisc_type)s is not supported; supported types: %(supported_qdisc_types)s.')