from __future__ import absolute_import, division, print_function
class AaaArgs(object):
    """The arg spec for the sonic_aaa module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'options': {'authentication': {'options': {'data': {'options': {'fail_through': {'type': 'bool'}, 'group': {'choices': ['ldap', 'radius', 'tacacs+'], 'type': 'str'}, 'local': {'type': 'bool'}}, 'type': 'dict'}}, 'type': 'dict'}}, 'type': 'dict'}, 'state': {'choices': ['merged', 'deleted', 'overridden', 'replaced'], 'default': 'merged', 'type': 'str'}}