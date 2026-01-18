from neutron_lib._i18n import _
from neutron_lib import exceptions
class FloatingIPPortAlreadyAssociated(exceptions.InUse):
    message = _('Cannot associate floating IP %(floating_ip_address)s (%(fip_id)s) with port %(port_id)s using fixed IP %(fixed_ip)s, as that fixed IP already has a floating IP on external network %(net_id)s.')