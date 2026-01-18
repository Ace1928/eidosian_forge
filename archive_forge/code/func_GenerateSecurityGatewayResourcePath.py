from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from urllib import parse
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import http_proxy_types
import socks
def GenerateSecurityGatewayResourcePath(project, location, sg_id):
    return 'projects/{}/iap_tunnel/locations/{}/destGroups/{}'.format(project, location, sg_id)