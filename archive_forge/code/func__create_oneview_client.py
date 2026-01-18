from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
def _create_oneview_client(self):
    if self.module.params.get('hostname'):
        config = dict(ip=self.module.params['hostname'], credentials=dict(userName=self.module.params['username'], password=self.module.params['password']), api_version=self.module.params['api_version'], image_streamer_ip=self.module.params['image_streamer_hostname'])
        self.oneview_client = OneViewClient(config)
    elif not self.module.params['config']:
        self.oneview_client = OneViewClient.from_environment_variables()
    else:
        self.oneview_client = OneViewClient.from_json_file(self.module.params['config'])