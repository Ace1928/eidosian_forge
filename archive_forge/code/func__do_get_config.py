import traceback
import lxml.etree
import ncclient
from os_ken.base import app_manager
from os_ken.lib.netconf import constants as nc_consts
from os_ken.lib import hub
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
from os_ken.lib.of_config import constants as ofc_consts
def _do_get_config(self, source):
    print('source = %s' % source)
    config_xml = self.switch.raw_get_config(source)
    tree = lxml.etree.fromstring(config_xml)
    self._validate(tree)