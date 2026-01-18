from __future__ import absolute_import, division, print_function
from ansible.module_utils.compat.paramiko import paramiko
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import get_logger
def _svc_disconnect(self):
    """
        Disconnect from the SSH server.
        """
    try:
        self.client.close()
        self.is_client_connected = False
        self.log('SSH disconnected')
        return True
    except Exception as e:
        self.log('SSH Disconnection failed %s', e)
        return False