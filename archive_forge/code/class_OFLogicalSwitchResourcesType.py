from os_ken.lib.of_config.base import _Base, _e, _ct
class OFLogicalSwitchResourcesType(_Base):
    _ELEMENTS = [_e('port', is_list=True), _e('queue', is_list=True), _e('certificate', is_list=False), _e('flow-table', is_list=True)]