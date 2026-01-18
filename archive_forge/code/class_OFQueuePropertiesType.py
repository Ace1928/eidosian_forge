from os_ken.lib.of_config.base import _Base, _e, _ct
class OFQueuePropertiesType(_Base):
    _ELEMENTS = [_e('min-rate', is_list=False), _e('max-rate', is_list=False), _e('experimenter', is_list=True)]