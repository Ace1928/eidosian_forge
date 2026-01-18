from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def _build_query_link(opts):
    query_link = '?marker={marker}&limit=10'
    return query_link