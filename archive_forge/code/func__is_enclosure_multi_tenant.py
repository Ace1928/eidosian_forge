from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def _is_enclosure_multi_tenant(self):
    """Determine if the enclosure is multi-tenant.

        The serial number of a multi-tenant enclosure will end in "-A" or "-B".

        :return: True/False if the enclosure is multi-tenant or not; None if unable to determine.
        """
    response = self.get_request(self.root_uri + self.service_root + 'Chassis/Enclosure')
    if response['ret'] is False:
        return None
    pattern = '.*-[A,B]'
    data = response['data']
    return re.match(pattern, data['SerialNumber']) is not None