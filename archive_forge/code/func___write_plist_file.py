from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def __write_plist_file(self, module, service_plist=None):
    if not service_plist:
        service_plist = {}
    if self.old_plistlib:
        plistlib.writePlist(service_plist, self.__file)
        return
    try:
        with open(self.__file, 'wb') as plist_fp:
            plistlib.dump(service_plist, plist_fp)
    except Exception as e:
        module.fail_json(msg='Failed to write to plist file  %s due to %s' % (self.__file, to_native(e)))