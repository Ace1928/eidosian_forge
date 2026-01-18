from __future__ import absolute_import, division, print_function
import os
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.urls import urlparse
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
Return a valid name for the package

        BIG-IP determines the package name by the content of the RPM info.
        It does not use the filename. Therefore, we do the same. This method
        is only used though when the file actually exists on your Ansible
        controller.

        If the package does not exist, then we instead use the filename
        portion of the 'package' argument that is provided.

        Non-existence typically occurs when using 'state' = 'absent'

        :return:
        