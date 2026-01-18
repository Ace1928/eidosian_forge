from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def __handle_param_enabled(self, module):
    if module.params['enabled'] is not None:
        service_plist = self.__read_plist_file(module)
        if module.params['enabled'] is not None:
            enabled = service_plist.get('RunAtLoad', False)
            if module.params['enabled'] != enabled:
                service_plist['RunAtLoad'] = module.params['enabled']
                if not module.check_mode:
                    self.__write_plist_file(module, service_plist)
                    self.__changed = True