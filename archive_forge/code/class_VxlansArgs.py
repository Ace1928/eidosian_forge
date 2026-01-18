from __future__ import absolute_import, division, print_function
class VxlansArgs(object):
    """The arg spec for the sonic_vxlans module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'evpn_nvo': {'type': 'str'}, 'name': {'required': True, 'type': 'str'}, 'source_ip': {'type': 'str'}, 'primary_ip': {'type': 'str'}, 'vlan_map': {'elements': 'dict', 'options': {'vlan': {'type': 'int'}, 'vni': {'required': True, 'type': 'int'}}, 'type': 'list'}, 'vrf_map': {'elements': 'dict', 'options': {'vni': {'required': True, 'type': 'int'}, 'vrf': {'type': 'str'}}, 'type': 'list'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}