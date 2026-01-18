import traceback
import lxml.etree
import ncclient
from os_ken.base import app_manager
from os_ken.lib.netconf import constants as nc_consts
from os_ken.lib import hub
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
from os_ken.lib.of_config import constants as ofc_consts
def _print_ports(self, tree, ns):
    for port in tree.findall('{%s}%s/{%s}%s' % (ns, ofc_consts.RESOURCES, ns, ofc_consts.PORT)):
        print(lxml.etree.tostring(port, pretty_print=True))