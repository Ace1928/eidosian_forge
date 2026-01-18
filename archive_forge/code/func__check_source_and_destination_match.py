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
def _check_source_and_destination_match(self):
    """Verify that destination and source are of the same IP version

        BIG-IP does not allow for mixing of the IP versions for destination and
        source addresses. For example, a destination IPv6 address cannot be
        associated with a source IPv4 address.

        This method checks that you specified the same IP version for these
        parameters.

        This method will not do this check if the virtual address name is used.

        Raises:
            F5ModuleError: Raised when the IP versions of source and destination differ.
        """
    if self.want.source and self.want.destination and (not self.want.destination_tuple.not_ip):
        want = ip_interface(u'{0}/{1}'.format(self.want.source_tuple.ip, self.want.source_tuple.cidr))
        have = ip_interface(u'{0}'.format(self.want.destination_tuple.ip))
        if want.version != have.version:
            raise F5ModuleError('The source and destination addresses for the virtual server must be be the same type (IPv4 or IPv6).')