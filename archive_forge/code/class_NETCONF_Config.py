from os_ken.lib import stringify
from .base import _Base, _ct, _e, _ns_netconf
from .generated_classes import *
class NETCONF_Config(_Base):
    _ELEMENTS = [_ct('capable-switch', OFCapableSwitchType, is_list=False)]

    def to_xml(self):
        return super(NETCONF_Config, self).to_xml('{%s}config' % _ns_netconf)