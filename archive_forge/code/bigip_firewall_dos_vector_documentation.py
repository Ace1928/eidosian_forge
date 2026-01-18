from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
Manages AFM DoS Profile Protocol SIP settings.

    Protocol SIP settings are a sub-collection attached to each profile.

    There are many similar vectors that can be managed here. This configuration
    is a sub-set of the device-config DoS vector configuration and excludes
    several attributes per-vector that are found in the device-config configuration.
    These include,

      * rateIncrease
      * rateLimit
      * rateThreshold
    