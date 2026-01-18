from __future__ import absolute_import, division, print_function
class L3_InterfacesArgs(object):

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'name': {'type': 'str', 'required': True}, 'ipv4': {'elements': 'dict', 'type': 'list', 'options': {'address': {'type': 'str'}, 'secondary': {'type': 'bool'}}}, 'ipv6': {'elements': 'dict', 'type': 'list', 'options': {'address': {'type': 'str'}}}}, 'type': 'list'}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'parsed', 'rendered'], 'default': 'merged', 'type': 'str'}}