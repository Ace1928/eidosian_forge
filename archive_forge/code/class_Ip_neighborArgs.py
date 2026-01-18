from __future__ import absolute_import, division, print_function
class Ip_neighborArgs(object):
    """The arg spec for the sonic_ip_neighbor module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'ipv4_arp_timeout': {'type': 'int'}, 'ipv4_drop_neighbor_aging_time': {'type': 'int'}, 'ipv6_drop_neighbor_aging_time': {'type': 'int'}, 'ipv6_nd_cache_expiry': {'type': 'int'}, 'num_local_neigh': {'type': 'int'}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted'], 'default': 'merged', 'type': 'str'}}