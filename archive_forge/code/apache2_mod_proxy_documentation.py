from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
 Returns members of the balancer as a generator object for later iteration.