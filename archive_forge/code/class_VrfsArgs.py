from __future__ import absolute_import, division, print_function
class VrfsArgs(object):
    """The arg spec for the sonic_vrfs module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'members': {'options': {'interfaces': {'elements': 'dict', 'options': {'name': {'type': 'str'}}, 'type': 'list'}}, 'type': 'dict'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'replaced', 'overridden', 'deleted'], 'default': 'merged', 'type': 'str'}}