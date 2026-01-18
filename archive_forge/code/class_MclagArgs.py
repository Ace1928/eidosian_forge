from __future__ import absolute_import, division, print_function
class MclagArgs(object):
    """The arg spec for the sonic_mclag module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'domain_id': {'required': True, 'type': 'int'}, 'gateway_mac': {'type': 'str'}, 'delay_restore': {'type': 'int'}, 'keepalive': {'type': 'int'}, 'peer_address': {'type': 'str'}, 'peer_link': {'type': 'str'}, 'members': {'options': {'portchannels': {'elements': 'dict', 'options': {'lag': {'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'peer_gateway': {'options': {'vlans': {'elements': 'dict', 'options': {'vlan': {'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'session_timeout': {'type': 'int'}, 'source_address': {'type': 'str'}, 'system_mac': {'type': 'str'}, 'unique_ip': {'options': {'vlans': {'elements': 'dict', 'options': {'vlan': {'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}