from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def _update_uri(self):
    return self.update_uri