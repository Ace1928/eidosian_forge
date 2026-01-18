from __future__ import absolute_import, division, print_function
import os
import time
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _auto_increment_hostname(count, hostname):
    """
    Allow a custom incremental count in the hostname when defined with the
    string formatting (%) operator. Otherwise, increment using name-01,
    name-02, name-03, and so forth.
    """
    if '%' not in hostname:
        hostname = '%s-%%01d' % hostname
    return [hostname % i for i in xrange(1, count + 1)]