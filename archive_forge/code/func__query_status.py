from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def _query_status():
    r = None
    try:
        r = client.get(url, timeout=timeout)
    except HwcClientException:
        return (None, '')
    try:
        s = navigate_value(r, ['status'])
        return (r, s)
    except Exception:
        return (None, '')