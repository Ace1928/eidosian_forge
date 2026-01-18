from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _get_base_perobject_url(self, mvalue):
    url_getting = self._get_basic_url(True)
    if not url_getting.endswith('}'):
        return url_getting
    last_token = url_getting.split('/')[-1]
    return url_getting.replace(last_token, str(mvalue))