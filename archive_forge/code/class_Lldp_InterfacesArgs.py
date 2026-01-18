from __future__ import absolute_import, division, print_function
class Lldp_InterfacesArgs(object):
    """The arg spec for the ios_lldp_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'name': {'required': True, 'type': 'str'}, 'transmit': {'type': 'bool'}, 'receive': {'type': 'bool'}, 'med_tlv_select': {'options': {'inventory_management': {'type': 'bool'}}, 'type': 'dict'}, 'tlv_select': {'options': {'power_management': {'type': 'bool'}}, 'type': 'dict'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'rendered', 'parsed', 'gathered'], 'default': 'merged', 'type': 'str'}}