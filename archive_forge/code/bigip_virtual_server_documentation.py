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
Verify that specified server type does not include forbidden profiles

        The type of the server determines the ``type``s of profiles that it accepts. This
        method checks that the server ``type`` that you specified is indeed one that can
        accept the profiles that you specified.

        The common situations are

        * ``standard`` types that include ``fasthttp``, ``fastl4``, or ``message routing`` profiles
        * ``fasthttp`` types that are missing a ``fasthttp`` profile
        * ``fastl4`` types that are missing a ``fastl4`` profile
        * ``message-routing`` types that are missing ``diameter`` or ``sip`` profiles

        Raises:
            F5ModuleError: Raised when a validation check fails.
        