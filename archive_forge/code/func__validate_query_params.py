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
def _validate_query_params(self, query_params):
    """
        Validate query_params that are passed in by users to make sure
        they're valid and return error if they're not valid.
        """
    invalid_query_params = []
    app = self._find_app(self.endpoint)
    nb_app = getattr(self.nb, app)
    nb_endpoint = getattr(nb_app, self.endpoint)
    base_url = self.nb.base_url
    junk, endpoint_url = nb_endpoint.url.split(base_url)
    response = open_url(base_url + '/docs/?format=openapi')
    try:
        raw_data = to_text(response.read(), errors='surrogate_or_strict')
    except UnicodeError:
        self._handle_errors(msg='Incorrect encoding of fetched payload from NetBox API.')
    try:
        openapi = json.loads(raw_data)
    except ValueError:
        self._handle_errors(msg='Incorrect JSON payload returned: %s' % raw_data)
    valid_query_params = openapi['paths'][endpoint_url + '/']['get']['parameters']
    for param in query_params:
        if param not in valid_query_params:
            invalid_query_params.append(param)
    if invalid_query_params:
        self._handle_errors('The following query_params are invalid: {0}'.format(', '.join(invalid_query_params)))