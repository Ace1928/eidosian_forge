from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _delete_instance(self):
    url = '{0}/{1}'.format(self.api_endpoint, self.name)
    if self.project:
        url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
    if not self.module.check_mode:
        self.client.do('DELETE', url)
    self.actions.append('delete')