from __future__ import absolute_import, division, print_function
class HostnameArgs(object):
    """The arg spec for the vyos_hostname module"""
    argument_spec = {'config': {'type': 'dict', 'options': {'hostname': {'type': 'str'}}}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted', 'gathered', 'parsed', 'rendered'], 'default': 'merged', 'type': 'str'}}