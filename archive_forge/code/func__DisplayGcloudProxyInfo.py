from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
import httplib2
def _DisplayGcloudProxyInfo(proxy_info, from_gcloud):
    """Displays Cloud SDK proxy information."""
    if not proxy_info:
        log.status.Print()
        return
    log.status.Print('Current effective Cloud SDK network proxy settings:')
    if not from_gcloud:
        log.status.Print("(These settings are from your machine's environment, not gcloud properties.)")
    proxy_type_name = http_proxy_types.REVERSE_PROXY_TYPE_MAP.get(proxy_info.proxy_type, 'UNKNOWN PROXY TYPE')
    log.status.Print('    type = {0}'.format(proxy_type_name))
    log.status.Print('    host = {0}'.format(proxy_info.proxy_host))
    log.status.Print('    port = {0}'.format(proxy_info.proxy_port))
    log.status.Print('    username = {0}'.format(encoding.Decode(proxy_info.proxy_user)))
    log.status.Print('    password = {0}'.format(encoding.Decode(proxy_info.proxy_pass)))
    log.status.Print()