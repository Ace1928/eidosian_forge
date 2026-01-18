from __future__ import absolute_import, division, print_function
import asyncio
import os
import urllib
from ansible.module_utils._text import to_native
from ansible.errors import AnsibleLookupError
from ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions import (
from ansible_collections.vmware.vmware_rest.plugins.module_utils.vmware_rest import (
def _filter_result(result):
    return [obj for obj in result if '%2f' not in obj['name']]