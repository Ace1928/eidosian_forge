from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def _set_root_uri(self, root_uris):
    """Set the root URI from a list of options.

        If the current root URI is good, just keep it.  Else cycle through our options until we find a good one.
        A URI is considered good if we can GET uri/redfish/v1.
        """
    for root_uri in root_uris:
        uri = root_uri + '/redfish/v1'
        response = self.get_request(uri)
        if response['ret']:
            self.root_uri = root_uri
            break