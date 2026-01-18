from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def _build_identity_object(all_opts):
    result = dict()
    result['id'] = None
    v = navigate_value(all_opts, ['ip_address'], None)
    result['ip_address'] = v
    v = navigate_value(all_opts, ['subnet_id'], None)
    result['subnet_id'] = v
    return result