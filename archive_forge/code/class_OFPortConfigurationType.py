from os_ken.lib.of_config.base import _Base, _e, _ct
class OFPortConfigurationType(_Base):
    _ELEMENTS = [_e('admin-state', is_list=False), _e('no-receive', is_list=False), _e('no-forward', is_list=False), _e('no-packet-in', is_list=False)]