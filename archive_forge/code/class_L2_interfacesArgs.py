from __future__ import absolute_import, division, print_function
class L2_interfacesArgs(object):
    """The arg spec for the junos_l2_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'access': {'type': 'dict', 'options': {'vlan': {'type': 'str'}}}, 'name': {'required': True, 'type': 'str'}, 'trunk': {'type': 'dict', 'options': {'allowed_vlans': {'elements': 'str', 'type': 'list'}, 'native_vlan': {'type': 'str'}}}, 'unit': {'type': 'int'}, 'enhanced_layer': {'type': 'bool'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}