import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
@staticmethod
def _get_tagging_asset_type(asset):
    objecttype = type(asset)
    if objecttype.__name__ in OBJECT_TO_TAGGING_ASSET_TYPE_MAP:
        return OBJECT_TO_TAGGING_ASSET_TYPE_MAP[objecttype.__name__]
    raise TypeError('Asset type %s cannot be tagged' % objecttype.__name__)