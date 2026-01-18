from __future__ import absolute_import, division, print_function
class Bgp_communitiesArgs(object):
    """The arg spec for the sonic_bgp_communities module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'aann': {'type': 'str'}, 'local_as': {'type': 'bool'}, 'match': {'choices': ['ALL', 'ANY'], 'default': 'ANY', 'type': 'str'}, 'members': {'options': {'regex': {'elements': 'str', 'type': 'list'}}, 'type': 'dict'}, 'name': {'required': True, 'type': 'str'}, 'no_advertise': {'type': 'bool'}, 'no_export': {'type': 'bool'}, 'no_peer': {'type': 'bool'}, 'permit': {'type': 'bool'}, 'type': {'choices': ['standard', 'expanded'], 'default': 'standard', 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'replaced', 'overridden'], 'default': 'merged', 'type': 'str'}}