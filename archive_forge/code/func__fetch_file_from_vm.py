from __future__ import absolute_import, division, print_function
import re
from os.path import exists, getsize
from socket import gaierror
from ssl import SSLError
from time import sleep
import traceback
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase
from ansible.module_utils.basic import missing_required_lib
def _fetch_file_from_vm(self, guestFilePath):
    try:
        fileTransferInformation = self.fileManager.InitiateFileTransferFromGuest(vm=self.vm, auth=self.vm_auth, guestFilePath=guestFilePath)
    except vim.fault.NoPermission as e:
        raise AnsibleError('No Permission Error: %s %s' % (to_native(e.msg), to_native(e.privilegeId)))
    except vmodl.fault.SystemError as e:
        if e.reason == 'vix error codes = (3016, 0).\n':
            raise AnsibleConnectionFailure('Connection failed, is the vm currently rebooting? Reason: %s' % to_native(e.reason))
        else:
            raise AnsibleConnectionFailure('Connection failed. Reason %s' % to_native(e.reason))
    except vim.fault.GuestOperationsUnavailable:
        raise AnsibleConnectionFailure('Cannot connect to guest. Native error: GuestOperationsUnavailable')
    url = self._fix_url_for_hosts(fileTransferInformation.url)
    response = requests.get(url, verify=self.validate_certs, stream=True)
    if response.status_code != 200:
        raise AnsibleError('Failed to fetch file')
    return response