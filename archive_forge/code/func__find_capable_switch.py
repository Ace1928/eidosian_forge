import ncclient
import ncclient.manager
import ncclient.xml_
from os_ken import exception as os_ken_exc
from os_ken.lib import of_config
from os_ken.lib.of_config import constants as ofc_consts
from os_ken.lib.of_config import classes as ofc
def _find_capable_switch(self, tree):
    capable_switch = None
    for element in tree:
        ns, tag = get_ns_tag(element.tag)
        if tag != ofc_consts.CAPABLE_SWITCH:
            continue
        assert capable_switch is None
        capable_switch = element
        if not self.version:
            versions = [(version, ns_) for version, ns_ in of_config.OFCONFIG_YANG_NAMESPACES.items() if ns == ns_]
            if versions:
                assert len(versions) == 1
                version = versions[0]
                self.version, self.namespace = version
    if not capable_switch:
        raise OFConfigCapableSwitchNotFound()
    return capable_switch