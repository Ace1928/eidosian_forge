from __future__ import absolute_import, division, print_function
class InterfacesArgs(object):
    """The arg spec for the vyos_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'description': {'type': 'str'}, 'duplex': {'choices': ['full', 'half', 'auto']}, 'enabled': {'default': True, 'type': 'bool'}, 'mtu': {'type': 'int'}, 'name': {'required': True, 'type': 'str'}, 'speed': {'choices': ['auto', '10', '100', '1000', '2500', '10000'], 'type': 'str'}, 'vifs': {'elements': 'dict', 'options': {'vlan_id': {'type': 'int'}, 'description': {'type': 'str'}, 'enabled': {'default': True, 'type': 'bool'}, 'mtu': {'type': 'int'}}, 'type': 'list'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'rendered', 'parsed', 'gathered'], 'default': 'merged', 'type': 'str'}}