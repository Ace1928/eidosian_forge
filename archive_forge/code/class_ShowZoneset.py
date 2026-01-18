from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class ShowZoneset(object):
    """docstring for ShowZoneset"""

    def __init__(self, module, vsan):
        self.vsan = vsan
        self.module = module
        self.zsDetails = {}
        self.parseCmdOutput()

    def execute_show_zoneset_cmd(self):
        command = 'show zoneset vsan ' + str(self.vsan)
        output = execute_show_command(command, self.module)[0]
        return output

    def parseCmdOutput(self):
        patZoneset = 'zoneset name (\\S+) vsan ' + str(self.vsan)
        patZone = 'zone name (\\S+) vsan ' + str(self.vsan)
        output = self.execute_show_zoneset_cmd().split('\n')
        for line in output:
            line = line.strip()
            mzs = re.match(patZoneset, line.strip())
            mz = re.match(patZone, line.strip())
            if mzs:
                zonesetname = mzs.group(1).strip()
                self.zsDetails[zonesetname] = []
                continue
            elif mz:
                zonename = mz.group(1).strip()
                v = self.zsDetails[zonesetname]
                v.append(zonename)
                self.zsDetails[zonesetname] = v

    def isZonesetPresent(self, zsname):
        return zsname in self.zsDetails.keys()

    def isZonePresentInZoneset(self, zsname, zname):
        if zsname in self.zsDetails.keys():
            return zname in self.zsDetails[zsname]
        return False