from __future__ import absolute_import, division, print_function
import string
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class showDeviceAliasDatabase(object):
    """docstring for showDeviceAliasDatabase"""

    def __init__(self, module):
        self.module = module
        self.da_dict = {}
        self.update()

    def execute_show_cmd(self, cmd):
        output = execute_show_command(cmd, self.module)[0]
        return output

    def update(self):
        command = 'show device-alias database'
        output = self.execute_show_cmd(command)
        self.da_list = output.split('\n')
        for eachline in self.da_list:
            if 'device-alias' in eachline:
                sv = eachline.strip().split()
                self.da_dict[sv[2]] = sv[4]

    def isNameInDaDatabase(self, name):
        return name in self.da_dict.keys()

    def isPwwnInDaDatabase(self, pwwn):
        newpwwn = ':'.join(['0' + str(ep) if len(ep) == 1 else ep for ep in pwwn.split(':')])
        return newpwwn in self.da_dict.values()

    def isNamePwwnPresentInDatabase(self, name, pwwn):
        newpwwn = ':'.join(['0' + str(ep) if len(ep) == 1 else ep for ep in pwwn.split(':')])
        if name in self.da_dict.keys():
            if newpwwn == self.da_dict[name]:
                return True
        return False

    def getPwwnByName(self, name):
        if name in self.da_dict.keys():
            return self.da_dict[name]
        else:
            return None

    def getNameByPwwn(self, pwwn):
        newpwwn = ':'.join(['0' + str(ep) if len(ep) == 1 else ep for ep in pwwn.split(':')])
        for n, p in self.da_dict.items():
            if p == newpwwn:
                return n
        return None