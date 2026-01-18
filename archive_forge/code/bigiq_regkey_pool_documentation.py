from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.teem import send_teem
Returns UUID of a given name

        Will search for a given name and return the first one returned to us. If no name,
        and therefore no ID, is found, will return the string "none". The string "none"
        is returned because if we were to return the None value, it would cause the
        license loading code to append a None string to the URI; essentially asking the
        remote device for its collection (which we dont want and which would cause the SDK
        to return an False error.

        :return:
        