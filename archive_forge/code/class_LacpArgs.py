from __future__ import absolute_import, division, print_function
class LacpArgs(object):
    """The arg spec for the junos_lacp module"""

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'type': 'dict', 'options': {'link_protection': {'choices': ['revertive', 'non-revertive'], 'type': 'str'}, 'system_priority': {'type': 'int'}}}, 'running_config': {'type': 'str'}, 'state': {'choices': ['merged', 'replaced', 'deleted', 'gathered', 'rendered', 'parsed'], 'default': 'merged', 'type': 'str'}}