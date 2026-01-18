from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import http.client
import logging
import select
import socket
import ssl
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as iap_utils
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def InitiateConnection(self):
    """Starts a tunnel to the destination through Security Gateway."""
    sg_utils.ValidateParameters(self._target)
    ca_certs = iap_utils.CheckCACertsFile(self._ignore_certs)
    if self._ignore_certs:
        ssl_ctx = ssl._create_unverified_context(cafile=ca_certs)
    else:
        ssl_ctx = ssl.create_default_context(cafile=ca_certs)
    proxy_host, proxy_port = sg_utils.GetProxyHostPort(self._target.url_override)
    conn = http.client.HTTPSConnection(proxy_host, proxy_port, context=ssl_ctx)
    dst_addr = '{}:{}'.format(self._target.host, self._target.port)
    headers = {}
    if callable(self._get_access_token_callback):
        headers['Proxy-Authorization'] = 'Bearer {}'.format(self._get_access_token_callback())
    headers['X-Resource-Key'] = sg_utils.GenerateSecurityGatewayResourcePath(self._target.project, self._target.region, self._target.security_gateway)
    log.debug('Sending headers: %s', headers)
    conn.request('CONNECT', dst_addr, headers=headers)
    resp = http.client.HTTPResponse(conn.sock, method='CONNECT', url=dst_addr)
    _, code, reason = resp._read_status()
    if code != http.client.OK:
        log.error('Connection request status [%s] with reason: %s', code, reason)
        raise SGConnectionError('Security Gateway failed to connect to destination url: ' + dst_addr)
    self._sock = conn.sock
    self._sock.setblocking(False)
    log.info('Connected to [%s]', dst_addr)
    self._sending_thread = threading.Thread(target=self._RunReceive)
    self._sending_thread.daemon = True
    self._sending_thread.start()