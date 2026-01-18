from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class LaunchCtlStop(LaunchCtlTask):

    def __init__(self, module, service, plist):
        super(LaunchCtlStop, self).__init__(module, service, plist)

    def runCommand(self):
        state, dummy, dummy, dummy = self.get_state()
        if state == ServiceState.STOPPED:
            if self._plist.is_changed():
                self.reload()
                self.stop()
        elif state in (ServiceState.STARTED, ServiceState.LOADED):
            if self._plist.is_changed():
                self.reload()
            self.stop()
        elif state == ServiceState.UNKNOWN:
            self.reload()
            self.stop()