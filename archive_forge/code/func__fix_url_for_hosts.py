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
def _fix_url_for_hosts(self, url):
    """
        Fix url if connection is a host.

        The host part of the URL is returned as '*' if the hostname to be used is the name of the server to which the call was made. For example, if the call is
        made to esx-svr-1.domain1.com, and the file is available for download from http://esx-svr-1.domain1.com/guestFile?id=1&token=1234, the URL returned may
        be http://*/guestFile?id=1&token=1234. The client replaces the asterisk with the server name on which it invoked the call.

        https://code.vmware.com/apis/358/vsphere#/doc/vim.vm.guest.FileManager.FileTransferInformation.html
        """
    return url.replace('*', self.vmware_host)