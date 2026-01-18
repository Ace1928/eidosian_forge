import os
import re
import time
import platform
import mimetypes
import subprocess
from os.path import join as pjoin
from collections import defaultdict
from libcloud.utils.py3 import ET, ensure_string
from libcloud.compute.base import Node, NodeState, NodeDriver
from libcloud.compute.types import Provider
from libcloud.utils.networking import is_public_subnet
def _cred_callback(self, cred, user_data):
    """
        Callback for the authentication scheme, which will provide username
        and password for the login. Reference: ( http://bit.ly/1U5yyQg )

        :param  cred: The credentials requested and the return
        :type   cred: ``list``

        :param  user_data: Custom data provided to the authentication routine
        :type   user_data: ``list``

        :rtype: ``int``
        """
    for credential in cred:
        if credential[0] == libvirt.VIR_CRED_AUTHNAME:
            credential[4] = self._key
        elif credential[0] == libvirt.VIR_CRED_PASSPHRASE:
            credential[4] = self._secret
    return 0