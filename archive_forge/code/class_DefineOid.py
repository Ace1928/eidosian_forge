from __future__ import absolute_import, division, print_function
import binascii
from collections import defaultdict
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
class DefineOid(object):

    def __init__(self, dotprefix=False):
        if dotprefix:
            dp = '.'
        else:
            dp = ''
        self.sysDescr = dp + '1.3.6.1.2.1.1.1.0'
        self.sysObjectId = dp + '1.3.6.1.2.1.1.2.0'
        self.sysUpTime = dp + '1.3.6.1.2.1.1.3.0'
        self.sysContact = dp + '1.3.6.1.2.1.1.4.0'
        self.sysName = dp + '1.3.6.1.2.1.1.5.0'
        self.sysLocation = dp + '1.3.6.1.2.1.1.6.0'
        self.ifIndex = dp + '1.3.6.1.2.1.2.2.1.1'
        self.ifDescr = dp + '1.3.6.1.2.1.2.2.1.2'
        self.ifMtu = dp + '1.3.6.1.2.1.2.2.1.4'
        self.ifSpeed = dp + '1.3.6.1.2.1.2.2.1.5'
        self.ifPhysAddress = dp + '1.3.6.1.2.1.2.2.1.6'
        self.ifAdminStatus = dp + '1.3.6.1.2.1.2.2.1.7'
        self.ifOperStatus = dp + '1.3.6.1.2.1.2.2.1.8'
        self.ifAlias = dp + '1.3.6.1.2.1.31.1.1.1.18'
        self.ipAdEntAddr = dp + '1.3.6.1.2.1.4.20.1.1'
        self.ipAdEntIfIndex = dp + '1.3.6.1.2.1.4.20.1.2'
        self.ipAdEntNetMask = dp + '1.3.6.1.2.1.4.20.1.3'