from __future__ import absolute_import, division, print_function
class Bgp_as_pathsArgs(object):
    """The arg spec for the sonic_bgp_as_paths module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'permit': {'required': False, 'type': 'bool'}, 'members': {'elements': 'str', 'required': False, 'type': 'list'}, 'name': {'required': True, 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}