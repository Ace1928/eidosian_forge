from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _launchctl(self, command):
    service_or_plist = self._plist.get_file() if command in ['load', 'unload'] else self._service if command in ['start', 'stop'] else ''
    rc, out, err = self._module.run_command('%s %s %s' % (self._launch, command, service_or_plist))
    if rc != 0:
        msg = "Unable to %s '%s' (%s): '%s'" % (command, self._service, self._plist.get_file(), err)
        self._module.fail_json(msg=msg)
    return (rc, out, err)