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
def _nb_endpoint_get(self, nb_endpoint, query_params, search_item):
    try:
        response = nb_endpoint.get(**query_params)
    except pynetbox.RequestError as e:
        self._handle_errors(msg=e.error)
    except ValueError:
        self._handle_errors(msg='More than one result returned for %s' % search_item)
    return response