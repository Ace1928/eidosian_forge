from os_ken.lib.of_config.base import _Base, _e, _ct
class OFCapableSwitchLogicalSwitchesType(_Base):
    _ELEMENTS = [_ct('switch', OFLogicalSwitchType, is_list=True)]