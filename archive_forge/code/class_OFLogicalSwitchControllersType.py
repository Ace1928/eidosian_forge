from os_ken.lib.of_config.base import _Base, _e, _ct
class OFLogicalSwitchControllersType(_Base):
    _ELEMENTS = [_ct('controller', OFControllerType, is_list=True)]