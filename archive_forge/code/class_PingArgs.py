from __future__ import absolute_import, division, print_function
class PingArgs(object):
    """The arg spec for the ios_ping module"""
    argument_spec = {'count': {'type': 'int'}, 'afi': {'choices': ['ip', 'ipv6'], 'default': 'ip', 'type': 'str'}, 'dest': {'required': True, 'type': 'str'}, 'df_bit': {'default': False, 'type': 'bool'}, 'source': {'type': 'str'}, 'size': {'type': 'int'}, 'ingress': {'type': 'str'}, 'egress': {'type': 'str'}, 'timeout': {'type': 'int'}, 'state': {'choices': ['absent', 'present'], 'default': 'present', 'type': 'str'}, 'vrf': {'type': 'str'}}