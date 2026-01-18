import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _version_from_url(url):
    if not url:
        return url
    url = urllib.parse.urlparse(url)
    for part in reversed(url.path.split('/')):
        try:
            if part[0] != 'v':
                continue
            return normalize_version_number(part)
        except Exception:
            pass
    return None