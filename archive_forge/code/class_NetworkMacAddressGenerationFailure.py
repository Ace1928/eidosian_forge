from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkMacAddressGenerationFailure(ServiceUnavailable):
    """An error related to MAC address generation on a network.

        :param net_id: The ID of the network MAC address generation failed on.
        """
    message = _('Unable to generate unique mac on network %(net_id)s.')