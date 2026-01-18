from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class Vsan(object):

    def __init__(self, vsanid):
        self.vsanid = vsanid
        self.vsanname = None
        self.vsanstate = None
        self.vsanoperstate = None
        self.vsaninterfaces = []