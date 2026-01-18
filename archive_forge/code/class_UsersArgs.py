from __future__ import absolute_import, division, print_function
class UsersArgs(object):
    """The arg spec for the sonic_users module
    """

    def __init__(self, **kwargs):
        pass
    argument_spec = {'config': {'elements': 'dict', 'options': {'name': {'required': True, 'type': 'str'}, 'password': {'type': 'str', 'no_log': True}, 'role': {'choices': ['admin', 'operator', 'netadmin', 'secadmin'], 'type': 'str'}, 'update_password': {'choices': ['always', 'on_create'], 'default': 'always', 'type': 'str'}}, 'type': 'list'}, 'state': {'choices': ['merged', 'deleted', 'overridden', 'replaced'], 'default': 'merged'}}