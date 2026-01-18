from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def _response_from_item(self, item):
    return remove_nones_from_dict({u'type': item.get(u'type')})