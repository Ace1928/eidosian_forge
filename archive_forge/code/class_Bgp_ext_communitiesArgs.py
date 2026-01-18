from __future__ import absolute_import, division, print_function
class Bgp_ext_communitiesArgs(object):
    """The arg spec for the sonic_bgp_ext_communities module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'match': {'choices': ['all', 'any'], 'default': 'any', 'type': 'str'}, 'members': {'mutually_exclusive': [['regex', 'route_origin'], ['regex', 'route_target']], 'options': {'regex': {'elements': 'str', 'type': 'list'}, 'route_origin': {'elements': 'str', 'type': 'list'}, 'route_target': {'elements': 'str', 'type': 'list'}}, 'type': 'dict'}, 'name': {'required': True, 'type': 'str'}, 'permit': {'type': 'bool'}, 'type': {'choices': ['standard', 'expanded'], 'default': 'standard', 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}