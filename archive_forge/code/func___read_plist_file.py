from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def __read_plist_file(self, module):
    service_plist = {}
    if self.old_plistlib:
        return plistlib.readPlist(self.__file)
    try:
        with open(self.__file, 'rb') as plist_fp:
            service_plist = plistlib.load(plist_fp)
    except Exception as e:
        module.fail_json(msg='Failed to read plist file %s due to %s' % (self.__file, to_native(e)))
    return service_plist