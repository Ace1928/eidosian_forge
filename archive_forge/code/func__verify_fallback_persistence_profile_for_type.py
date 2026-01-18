from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from collections import namedtuple
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.constants import (
from ..module_utils.compare import cmp_simple_list
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import (
from ..module_utils.teem import send_teem
def _verify_fallback_persistence_profile_for_type(self):
    """Verify that the server type supports fallback persistence profiles

        Verifies that the specified server type supports fallback persistence profiles.
        Some virtual servers do not support these types of profiles. This method will
        check that the type actually supports what you are sending it.

        Types that do not, at this time, support fallback persistence profiles include,

        * dhcp
        * message-routing
        * reject
        * stateless
        * forwarding-ip
        * forwarding-l2
        * performance-http

        Raises:
            F5ModuleError: Raised if server type does not support fallback persistence profiles.
        """
    default_profile_not_allowed = ['dhcp', 'message-routing', 'reject', 'stateless', 'forwarding-ip', 'forwarding-l2', 'performance-http']
    if self.want.ip_protocol in default_profile_not_allowed:
        raise F5ModuleError("The '{0}' server type does not support a 'fallback_persistence_profile'".format(self.want.type))