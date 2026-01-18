from os_ken.lib.of_config.base import _Base, _e, _ct
class OFControllerType(_Base):
    _ELEMENTS = [_e('id', is_list=False), _e('role', is_list=False), _e('ip-address', is_list=False), _e('port', is_list=False), _e('local-ip-address', is_list=False), _e('local-port', is_list=False), _e('protocol', is_list=False), _ct('state', OFControllerStateType, is_list=False)]