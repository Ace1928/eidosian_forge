from __future__ import absolute_import, division, print_function
import time
import ssl
from datetime import datetime
from ansible.module_utils.six.moves.urllib.error import URLError
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _set_volume_url(self, item):
    path = urlparse(item['selfLink']).path
    self.volume_url = 'https://{0}:{1}{2}'.format(self.client.provider['server'], self.client.provider['server_port'], path)