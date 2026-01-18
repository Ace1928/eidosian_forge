from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class Plist:

    def __init__(self, module, service):
        self.__changed = False
        self.__service = service
        state, pid, dummy, dummy = LaunchCtlList(module, self.__service).run()
        self.old_plistlib = hasattr(plistlib, 'readPlist')
        self.__file = self.__find_service_plist(self.__service)
        if self.__file is None:
            msg = 'Unable to infer the path of %s service plist file' % self.__service
            if pid is None and state == ServiceState.UNLOADED:
                msg += ' and it was not found among active services'
            module.fail_json(msg=msg)
        self.__update(module)

    @staticmethod
    def __find_service_plist(service_name):
        """Finds the plist file associated with a service"""
        launchd_paths = [os.path.join(os.getenv('HOME'), 'Library/LaunchAgents'), '/Library/LaunchAgents', '/Library/LaunchDaemons', '/System/Library/LaunchAgents', '/System/Library/LaunchDaemons']
        for path in launchd_paths:
            try:
                files = os.listdir(path)
            except OSError:
                continue
            filename = '%s.plist' % service_name
            if filename in files:
                return os.path.join(path, filename)
        return None

    def __update(self, module):
        self.__handle_param_enabled(module)
        self.__handle_param_force_stop(module)

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

    def is_changed(self):
        return self.__changed

    def get_file(self):
        return self.__file