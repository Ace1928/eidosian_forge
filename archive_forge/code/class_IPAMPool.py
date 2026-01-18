from .. import errors
from ..utils import normalize_links, version_lt
class IPAMPool(dict):
    """
    Create an IPAM pool config dictionary to be added to the
    ``pool_configs`` parameter of
    :py:class:`~docker.types.IPAMConfig`.

    Args:

        subnet (str): Custom subnet for this IPAM pool using the CIDR
            notation. Defaults to ``None``.
        iprange (str): Custom IP range for endpoints in this IPAM pool using
            the CIDR notation. Defaults to ``None``.
        gateway (str): Custom IP address for the pool's gateway.
        aux_addresses (dict): A dictionary of ``key -> ip_address``
            relationships specifying auxiliary addresses that need to be
            allocated by the IPAM driver.

    Example:

        >>> ipam_pool = docker.types.IPAMPool(
            subnet='124.42.0.0/16',
            iprange='124.42.0.0/24',
            gateway='124.42.0.254',
            aux_addresses={
                'reserved1': '124.42.1.1'
            }
        )
        >>> ipam_config = docker.types.IPAMConfig(
                pool_configs=[ipam_pool])
    """

    def __init__(self, subnet=None, iprange=None, gateway=None, aux_addresses=None):
        self.update({'Subnet': subnet, 'IPRange': iprange, 'Gateway': gateway, 'AuxiliaryAddresses': aux_addresses})