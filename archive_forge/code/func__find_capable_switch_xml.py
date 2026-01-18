import ncclient
import ncclient.manager
import ncclient.xml_
from os_ken import exception as os_ken_exc
from os_ken.lib import of_config
from os_ken.lib.of_config import constants as ofc_consts
from os_ken.lib.of_config import classes as ofc
def _find_capable_switch_xml(self, tree):
    return ncclient.xml_.to_xml(self._find_capable_switch(tree))