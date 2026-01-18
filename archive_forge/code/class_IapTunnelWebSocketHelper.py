from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import ssl
import sys
import threading
import traceback
from googlecloudsdk.api_lib.compute import iap_tunnel_lightweight_websocket as iap_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
import six
import websocket
class IapTunnelWebSocketHelper(object):
    """Helper class for common operations on websocket and related metadata."""

    def __init__(self, url, headers, ignore_certs, proxy_info, on_data, on_close, should_use_new_websocket, conn_id=0):
        self._on_data = on_data
        self._on_close = on_close
        self._proxy_info = proxy_info
        self._conn_id = conn_id
        self._receiving_thread = None
        ca_certs = utils.CheckCACertsFile(ignore_certs)
        self._sslopt = {'cert_reqs': ssl.CERT_REQUIRED, 'ca_certs': ca_certs}
        if ignore_certs:
            self._sslopt['cert_reqs'] = ssl.CERT_NONE
            self._sslopt['check_hostname'] = False
        caa_config = context_aware.Config()
        if caa_config:
            cert_path = caa_config.encrypted_client_cert_path
            log.debug('Using client certificate %s', cert_path)
            self._sslopt['certfile'] = cert_path
            self._sslopt['password'] = caa_config.encrypted_client_cert_password
        if log.GetVerbosity() != logging.DEBUG:
            logging.getLogger('websocket').setLevel(logging.CRITICAL)
        self._is_closed = False
        self._error_msg = ''
        self._should_use_new_websocket = should_use_new_websocket
        if self._should_use_new_websocket:
            self._websocket = iap_websocket.IapLightWeightWebsocket(url, header=headers, on_close=self._OnClose, on_data=self._OnData, on_error=self._OnError, subprotocols=[utils.SUBPROTOCOL_NAME])
        else:
            self._websocket = websocket.WebSocketApp(url, header=headers, on_close=self._OnClose, on_data=self._OnData, on_error=self._OnError, subprotocols=[utils.SUBPROTOCOL_NAME])

    def __del__(self):
        self.Close()

    def Close(self, msg=''):
        """Close the WebSocket."""
        if not self._is_closed:
            try:
                self._websocket.close()
            except:
                pass
            if not self._error_msg:
                self._error_msg = msg
            self._is_closed = True

    def IsClosed(self):
        """Check to see if WebSocket has closed."""
        return self._is_closed or (self._receiving_thread and (not self._receiving_thread.is_alive()))

    def ErrorMsg(self):
        return self._error_msg

    def Send(self, send_data):
        """Send data on WebSocket connection."""
        try:
            if log.GetVerbosity() == logging.DEBUG:
                log.debug('[%d] SEND data_len [%d] send_data[:20] %r', self._conn_id, len(send_data), send_data[:20])
            self._websocket.send(send_data, opcode=websocket.ABNF.OPCODE_BINARY)
        except EnvironmentError:
            self.Close()
            raise
        except websocket.WebSocketConnectionClosedException:
            self.Close()
            raise WebSocketConnectionClosed()
        except Exception as e:
            log.debug('[%d] Error during WebSocket send of Data message.', self._conn_id, exc_info=True)
            tb = sys.exc_info()[2]
            self.Close()
            exceptions.reraise(WebSocketSendError(traceback.format_exception_only(type(e), e), tb=tb))

    def SendClose(self):
        """Send WebSocket Close message if possible."""
        if self._should_use_new_websocket:
            sock = self._websocket
        else:
            sock = self._websocket.sock
        if sock:
            log.debug('[%d] CLOSE', self._conn_id)
            try:
                sock.send_close()
            except (EnvironmentError, websocket.WebSocketConnectionClosedException) as e:
                log.info('[%d] Unable to send WebSocket Close message [%s].', self._conn_id, six.text_type(e))
                self.Close()
            except:
                log.info('[%d] Error during WebSocket send of Close message.', self._conn_id, exc_info=True)
                self.Close()

    def StartReceivingThread(self):
        if not self._is_closed:
            self._receiving_thread = threading.Thread(target=self._ReceiveFromWebSocket)
            self._receiving_thread.daemon = True
            self._receiving_thread.start()

    def _OnClose(self, close_code, close_reason):
        """Callback for WebSocket Close messages."""
        if close_code is None and close_reason is None:
            self.Close()
            return
        close_msg = '%r: %r' % (close_code, close_reason)
        log.info('[%d] Received WebSocket Close message [%s].', self._conn_id, close_msg)
        self.Close(msg=close_msg)
        if close_code == 4004:
            return
        try:
            self._on_close()
        except (EnvironmentError, exceptions.Error):
            log.info('[%d] Error while processing Close message', self._conn_id, exc_info=True)
            raise

    def _OnData(self, binary_data, opcode, unused_finished=0):
        """Callback for WebSocket Data messages."""
        if log.GetVerbosity() == logging.DEBUG:
            log.debug('[%d] RECV opcode [%r] data_len [%d] binary_data[:20] [%r]', self._conn_id, opcode, len(binary_data), binary_data[:20])
        try:
            if opcode not in (websocket.ABNF.OPCODE_CONT, websocket.ABNF.OPCODE_BINARY):
                raise WebSocketInvalidOpcodeError('Unexpected WebSocket opcode [%r].' % opcode)
            self._on_data(binary_data)
        except EnvironmentError as e:
            log.info('[%d] Error [%s] while sending to client.', self._conn_id, six.text_type(e))
            self.Close()
            raise
        except:
            log.info('[%d] Error while processing Data message.', self._conn_id, exc_info=True)
            self.Close()
            raise

    def _OnError(self, exception_obj):
        if not self._is_closed:
            log.debug('[%d] Error during WebSocket processing.', self._conn_id, exc_info=True)
            log.info('[%d] Error during WebSocket processing:\n' + ''.join(traceback.format_exception_only(type(exception_obj), exception_obj)), self._conn_id)
            self._error_msg = six.text_type(exception_obj)

    def _ReceiveFromWebSocket(self):
        """Receive data from WebSocket connection."""
        try:
            if self._proxy_info:
                http_proxy_auth = None
                if self._proxy_info.proxy_user or self._proxy_info.proxy_pass:
                    http_proxy_auth = (encoding.Decode(self._proxy_info.proxy_user), encoding.Decode(self._proxy_info.proxy_pass))
                self._websocket.run_forever(origin=TUNNEL_CLOUDPROXY_ORIGIN, sslopt=self._sslopt, http_proxy_host=self._proxy_info.proxy_host, http_proxy_port=self._proxy_info.proxy_port, http_proxy_auth=http_proxy_auth)
            else:
                self._websocket.run_forever(origin=TUNNEL_CLOUDPROXY_ORIGIN, sslopt=self._sslopt)
        except:
            try:
                log.info('[%d] Error while receiving from WebSocket.', self._conn_id, exc_info=True)
            except:
                pass
        try:
            self.Close()
        except:
            try:
                log.info('[%d] Error while closing in receiving thread.', self._conn_id, exc_info=True)
            except:
                pass