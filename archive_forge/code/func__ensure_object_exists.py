from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _ensure_object_exists(self, nb_endpoint, endpoint_name, name, data):
    """Used when `state` is present to make sure object exists or if the object exists
        that it is updated
        :params nb_endpoint (pynetbox endpoint object): This is the nb endpoint to be used
        to create or update the object
        :params endpoint_name (str): Endpoint name that was created/updated. ex. device
        :params name (str): Name of the object
        :params data (dict): User defined data passed into the module
        """
    if not self.nb_object:
        self.nb_object, diff = self._create_netbox_object(nb_endpoint, data)
        self.result['msg'] = '%s %s created' % (endpoint_name, name)
        self.result['changed'] = True
        self.result['diff'] = diff
    else:
        self.nb_object, diff = self._update_netbox_object(data)
        if self.nb_object is False:
            self._handle_errors(msg="Request failed, couldn't update device: %s" % name)
        if diff:
            self.result['msg'] = '%s %s updated' % (endpoint_name, name)
            self.result['changed'] = True
            self.result['diff'] = diff
        else:
            self.result['msg'] = '%s %s already exists' % (endpoint_name, name)