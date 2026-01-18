from __future__ import absolute_import, division, print_function
class Lacp_interfacesArgs(object):
    """The arg spec for the junos_lacp_interfaces module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'force_up': {'type': 'bool'}, 'name': {'type': 'str'}, 'period': {'choices': ['fast', 'slow']}, 'port_priority': {'type': 'int'}, 'sync_reset': {'choices': ['disable', 'enable'], 'type': 'str'}, 'system': {'options': {'mac': {'type': 'dict', 'options': {'address': {'type': 'str'}}}, 'priority': {'type': 'int'}}, 'type': 'dict'}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}