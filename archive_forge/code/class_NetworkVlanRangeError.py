from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkVlanRangeError(NeutronException):
    message = _("Invalid network VLAN range: '%(vlan_range)s' - '%(error)s'.")

    def __init__(self, **kwargs):
        if isinstance(kwargs['vlan_range'], tuple):
            kwargs['vlan_range'] = '%d:%d' % kwargs['vlan_range']
        super().__init__(**kwargs)