from __future__ import absolute_import, division, print_function
class Acl_interfacesArgs(object):
    """The arg spec for the junos_acl_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'access_groups': {'elements': 'dict', 'options': {'acls': {'elements': 'dict', 'options': {'direction': {'choices': ['in', 'out'], 'type': 'str'}, 'name': {'type': 'str'}}, 'type': 'list'}, 'afi': {'choices': ['ipv4', 'ipv6'], 'type': 'str'}}, 'type': 'list'}, 'name': {'type': 'str'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}