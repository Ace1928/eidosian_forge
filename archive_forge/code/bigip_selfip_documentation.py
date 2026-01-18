from __future__ import absolute_import, division, print_function
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ipaddress import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.ipaddress import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
Returns services formatted for consumption by f5-sdk update

        The BIG-IP endpoint for services takes different values depending on
        what you want the "allowed services" to be. It can be any of the
        following

            - a list containing "protocol:port" values
            - the string "all"
            - a null value, or None

        This is a convenience function to massage the values the user has
        supplied so that they are formatted in such a way that BIG-IP will
        accept them and apply the specified policy.
        