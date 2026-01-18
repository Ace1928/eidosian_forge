from __future__ import absolute_import, division, print_function
import atexit
import time
import re
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.ansible_release import __version__ as ANSIBLE_VERSION
class XenServerObject(object):
    """Base class for all XenServer objects.

    This class contains active XAPI session reference and common
    attributes with useful info about XenServer host/pool.

    Attributes:
        module: Reference to Ansible module object.
        xapi_session: Reference to XAPI session.
        pool_ref (str): XAPI reference to a pool currently connected to.
        default_sr_ref (str): XAPI reference to a pool default
            Storage Repository.
        host_ref (str): XAPI rerefence to a host currently connected to.
        xenserver_version (list of str): Contains XenServer major and
            minor version.
    """

    def __init__(self, module):
        """Inits XenServerObject using common module parameters.

        Args:
            module: Reference to Ansible module object.
        """
        if not HAS_XENAPI:
            module.fail_json(changed=False, msg=missing_required_lib('XenAPI'), exception=XENAPI_IMP_ERR)
        self.module = module
        self.xapi_session = XAPI.connect(module)
        try:
            self.pool_ref = self.xapi_session.xenapi.pool.get_all()[0]
            self.default_sr_ref = self.xapi_session.xenapi.pool.get_default_SR(self.pool_ref)
            self.xenserver_version = get_xenserver_version(module)
        except XenAPI.Failure as f:
            self.module.fail_json(msg='XAPI ERROR: %s' % f.details)