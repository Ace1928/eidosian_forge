from __future__ import absolute_import, division, print_function
class VlansArgs(object):
    """The arg spec for the junos_vlans module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'description': {}, 'name': {'required': True, 'type': 'str'}, 'vlan_id': {'type': 'int'}, 'l3_interface': {'type': 'str'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}