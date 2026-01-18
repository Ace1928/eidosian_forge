from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def __handle_param_force_stop(self, module):
    if module.params['force_stop'] is not None:
        service_plist = self.__read_plist_file(module)
        if module.params['force_stop'] is not None:
            keep_alive = service_plist.get('KeepAlive', False)
            if module.params['force_stop'] and keep_alive:
                service_plist['KeepAlive'] = not module.params['force_stop']
                if not module.check_mode:
                    self.__write_plist_file(module, service_plist)
                    self.__changed = True