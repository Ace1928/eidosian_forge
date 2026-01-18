from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import (
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils import deps
def _get_existing_dbus_session(self):
    """
        Detects and returns an existing D-Bus session bus address.

        :returns: string -- D-Bus session bus address. If a running D-Bus session was not detected, returns None.
        """
    uid = os.getuid()
    self.module.debug('Trying to detect existing D-Bus user session for user: %d' % uid)
    for pid in psutil.pids():
        try:
            process = psutil.Process(pid)
            process_real_uid, dummy, dummy = process.uids()
            if process_real_uid == uid and 'DBUS_SESSION_BUS_ADDRESS' in process.environ():
                dbus_session_bus_address_candidate = process.environ()['DBUS_SESSION_BUS_ADDRESS']
                self.module.debug('Found D-Bus user session candidate at address: %s' % dbus_session_bus_address_candidate)
                dbus_send_cmd = self.module.get_bin_path('dbus-send', required=True)
                command = [dbus_send_cmd, '--address=%s' % dbus_session_bus_address_candidate, '--type=signal', '/', 'com.example.test']
                rc, dummy, dummy = self.module.run_command(command)
                if rc == 0:
                    self.module.debug('Verified D-Bus user session candidate as usable at address: %s' % dbus_session_bus_address_candidate)
                    return dbus_session_bus_address_candidate
        except psutil.AccessDenied:
            pass
        except psutil.NoSuchProcess:
            pass
    self.module.debug('Failed to find running D-Bus user session, will use dbus-run-session')
    return None