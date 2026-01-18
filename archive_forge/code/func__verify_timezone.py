from __future__ import absolute_import, division, print_function
import errno
import os
import platform
import random
import re
import string
import filecmp
from ansible.module_utils.basic import AnsibleModule, get_distribution
from ansible.module_utils.six import iteritems
def _verify_timezone(self):
    tz = self.value['name']['planned']
    out = self.execute(self.systemsetup, '-listtimezones').splitlines()[1:]
    tz_list = list(map(lambda x: x.strip(), out))
    if tz not in tz_list:
        self.abort('given timezone "%s" is not available' % tz)
    return tz