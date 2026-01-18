from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.six import iteritems
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
def _fail_json(msg):
    """Replace the AnsibleModule fail_json here
    :param msg: The message for the failure
    :type msg: str
    """
    raise Exception(msg)