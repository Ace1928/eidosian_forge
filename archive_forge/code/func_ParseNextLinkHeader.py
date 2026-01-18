from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import re
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_creds as v2_2_creds
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def ParseNextLinkHeader(resp):
    """Returns "next" link from RFC 5988 Link header or None if not present."""
    link = resp.get('link')
    if not link:
        return None
    m = re.match('.*<(.+)>;\\s*rel="next".*', link)
    if not m:
        return None
    return m.group(1)