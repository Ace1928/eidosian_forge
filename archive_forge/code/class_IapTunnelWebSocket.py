from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import logging
import threading
import time
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_helper as helper
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
import six
from six.moves import queue
class IapTunnelWebSocket(object):
    """Cloud IAP WebSocket class for tunnelling connections.

  It takes in local data (via Send()) which it sends over the websocket. It
  takes data from the websocket and gives it to data_handler_callback.
  """

    def __init__(self, tunnel_target, get_access_token_callback, data_handler_callback, close_handler_callback, user_agent, conn_id=0, ignore_certs=False):
        self._tunnel_target = tunnel_target
        self._get_access_token_callback = get_access_token_callback
        self._data_handler_callback = data_handler_callback
        self._close_handler_callback = close_handler_callback
        self._ignore_certs = ignore_certs
        self._user_agent = user_agent
        self._websocket_helper = None
        self._connect_msg_received = False
        self._connection_sid = None
        self._stopping = False
        self._close_message_sent = False
        self._send_and_reconnect_thread = None
        self._input_eof = False
        self._sent_all = threading.Event()
        self._cant_send_ack = threading.Event()
        self._total_bytes_confirmed = 0
        self._total_bytes_received = 0
        self._total_bytes_received_and_acked = 0
        self._unsent_data = queue.Queue(maxsize=MAX_UNSENT_QUEUE_LENGTH)
        self._unconfirmed_data = collections.deque()
        self._data_to_resend = queue.Queue()
        self._conn_id = conn_id

    def __del__(self):
        if self._websocket_helper:
            self._websocket_helper.Close()

    def Close(self):
        """Close down local connection and WebSocket connection."""
        self._stopping = True
        self._unsent_data.put(StoppingError)
        try:
            self._close_handler_callback()
        except:
            pass
        if self._websocket_helper:
            if not self._close_message_sent:
                self._websocket_helper.SendClose()
                self._close_message_sent = True
            self._websocket_helper.Close()

    def InitiateConnection(self):
        """Initiate the WebSocket connection."""
        utils.CheckPythonVersion(self._ignore_certs)
        utils.ValidateParameters(self._tunnel_target)
        self._StartNewWebSocket()
        self._WaitForOpenOrRaiseError()
        self._send_and_reconnect_thread = threading.Thread(target=self._SendDataAndReconnectWebSocket)
        self._send_and_reconnect_thread.daemon = True
        self._send_and_reconnect_thread.start()

    def Send(self, bytes_to_send):
        """Send bytes over WebSocket connection.

    Args:
      bytes_to_send: The bytes to send. Must not be empty.

    Raises:
      ConnectionReconnectTimeout: If something is preventing data from being
        sent.
    """
        while bytes_to_send:
            first_to_send = bytes_to_send[:utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE]
            bytes_to_send = bytes_to_send[utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE:]
            if first_to_send:
                self._EnqueueBytesWithWaitForReconnect(first_to_send)

    def LocalEOF(self):
        """Indicate that the local input gave an EOF.

    This should always be called after finishing sending data, as to stop the
    sending thread.
    """
        self._unsent_data.put(EOFError)

    def WaitForAllSent(self):
        """Wait until all local data has been sent on the websocket.

    Blocks until either all data from Send() has been sent, or it times out
    waiting. Once true, always returns true. Even if this returns true, a
    reconnect could occur causing previously sent data to be resent. Must only
    be called after an EOF has been given to Send().

    Returns:
      True on success, False on timeout.
    """
        return self._sent_all.wait(ALL_DATA_SENT_WAIT_TIME_SEC)

    def _AttemptReconnect(self, reconnect_func):
        """Attempt to reconnect with a new WebSocket."""
        r = retry.Retryer(max_wait_ms=MAX_RECONNECT_WAIT_TIME_MS, exponential_sleep_multiplier=1.1, wait_ceiling_ms=MAX_RECONNECT_SLEEP_TIME_MS)
        try:
            r.RetryOnException(func=reconnect_func, sleep_ms=RECONNECT_INITIAL_SLEEP_MS)
        except retry.RetryException:
            log.warning('[%d] Unable to reconnect within [%d] ms', self._conn_id, MAX_RECONNECT_WAIT_TIME_MS, exc_info=True)
            self._StopConnectionAsync()

    def _EnqueueBytesWithWaitForReconnect(self, bytes_to_send):
        """Add bytes to the queue; block waiting for reconnect if queue is full.

    Args:
      bytes_to_send: The local bytes to send over the websocket. At most
        utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE.

    Raises:
      ConnectionReconnectTimeout: If something is preventing data from being
        sent.
      ConnectionCreationError: If the connection was closed and no more
        reconnect retries will be performed.
    """
        end_time = time.time() + MAX_RECONNECT_WAIT_TIME_MS / 1000.0
        while time.time() < end_time and (not self._stopping):
            try:
                self._unsent_data.put(bytes_to_send, timeout=MAX_WEBSOCKET_SEND_WAIT_TIME_SEC)
                if log.GetVerbosity() == logging.DEBUG:
                    log.debug('[%d] ENQUEUED data_len [%d] bytes_to_send[:20] [%r]', self._conn_id, len(bytes_to_send), bytes_to_send[:20])
                return
            except queue.Full:
                pass
        if self._stopping:
            raise ConnectionCreationError('Unexpected error while reconnecting. Check logs for more details.')
        raise ConnectionReconnectTimeout()

    def _HasConnected(self):
        """Returns true if we received a connect message."""
        return self._connect_msg_received

    def _IsClosed(self):
        return self._websocket_helper and self._websocket_helper.IsClosed() or (self._send_and_reconnect_thread and (not self._send_and_reconnect_thread.is_alive()))

    def _StartNewWebSocket(self):
        """Start a new WebSocket and thread to listen for incoming data."""
        headers = ['User-Agent: ' + self._user_agent]
        log.debug('[%d] user-agent [%s]', self._conn_id, self._user_agent)
        request_reason = properties.VALUES.core.request_reason.Get()
        if request_reason:
            headers += ['X-Goog-Request-Reason: ' + request_reason]
        if self._get_access_token_callback:
            headers += ['Authorization: Bearer ' + self._get_access_token_callback()]
        log.debug('[%d] Using new websocket library', self._conn_id)
        if self._connection_sid:
            url = utils.CreateWebSocketReconnectUrl(self._tunnel_target, self._connection_sid, self._total_bytes_received, should_use_new_websocket=True)
            log.info('[%d] Reconnecting with URL [%r]', self._conn_id, url)
        else:
            url = utils.CreateWebSocketConnectUrl(self._tunnel_target, should_use_new_websocket=True)
            log.info('[%d] Connecting with URL [%r]', self._conn_id, url)
        self._connect_msg_received = False
        self._websocket_helper = helper.IapTunnelWebSocketHelper(url, headers, self._ignore_certs, self._tunnel_target.proxy_info, self._OnData, self._OnClose, should_use_new_websocket=True, conn_id=self._conn_id)
        self._websocket_helper.StartReceivingThread()

    def _SendAck(self):
        """Send an ACK back to server."""
        if self._total_bytes_received > self._total_bytes_received_and_acked:
            bytes_received = self._total_bytes_received
            try:
                ack_data = utils.CreateSubprotocolAckFrame(bytes_received)
                self._websocket_helper.Send(ack_data)
                self._total_bytes_received_and_acked = bytes_received
            except helper.WebSocketConnectionClosed:
                raise
            except EnvironmentError as e:
                log.info('[%d] Unable to send WebSocket ack [%s]', self._conn_id, six.text_type(e))
            except:
                if not self._IsClosed():
                    log.info('[%d] Error while attempting to ack [%d] bytes', self._conn_id, bytes_received, exc_info=True)
                else:
                    raise
            finally:
                self._cant_send_ack.clear()

    def _MaybeSendAck(self):
        """Decide if an ACK should be sent back to the server."""
        if self._cant_send_ack.is_set():
            return
        total_bytes = self._total_bytes_received
        bytes_recv_and_ackd = self._total_bytes_received_and_acked
        window_size = utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE
        if total_bytes - bytes_recv_and_ackd > 2 * window_size:
            self._cant_send_ack.set()
            self._unsent_data.put(SendAckNotification)

    def _SendDataAndReconnectWebSocket(self):
        """Main function for send_and_reconnect_thread."""

        def SendData():
            if not self._stopping:
                self._SendQueuedData()
                self._SendAck()

        def Reconnect():
            if not self._stopping:
                self._StartNewWebSocket()
                self._WaitForOpenOrRaiseError()
        try:
            while not self._stopping:
                try:
                    SendData()
                except Exception as e:
                    log.debug('[%d] Error while sending data, trying to reconnect [%s]', self._conn_id, six.text_type(e))
                    self._AttemptReconnect(Reconnect)
        finally:
            self.Close()

    def _SendQueuedData(self):
        """Send data that is sitting in the unsent data queue."""
        try:
            while not self._stopping:
                self._MaybeSendAck()
                try:
                    if not self._data_to_resend.empty():
                        data = self._data_to_resend.get()
                    else:
                        data = self._unsent_data.get(timeout=MAX_WEBSOCKET_SEND_WAIT_TIME_SEC)
                except queue.Empty:
                    if self._IsClosed():
                        raise helper.WebSocketConnectionClosed
                    break
                if data is EOFError or data is StoppingError:
                    self._stopping = True
                    if data is EOFError:
                        self._input_eof = True
                    break
                if data is SendAckNotification:
                    self._SendAck()
                    continue
                self._unconfirmed_data.append(data)
                send_data = utils.CreateSubprotocolDataFrame(data)
                self._websocket_helper.Send(send_data)
        finally:
            if self._input_eof and self._data_to_resend.empty() and self._unsent_data.empty():
                self._sent_all.set()

    def _StopConnectionAsync(self):
        self._stopping = True

    def _WaitForOpenOrRaiseError(self):
        """Wait for WebSocket open confirmation or any error condition."""
        for _ in range(MAX_WEBSOCKET_OPEN_WAIT_TIME_SEC * 100):
            if self._IsClosed():
                break
            if self._HasConnected():
                return
            time.sleep(0.01)
        if self._websocket_helper and self._websocket_helper.IsClosed() and self._websocket_helper.ErrorMsg():
            extra_msg = ''
            if self._websocket_helper.ErrorMsg().startswith('Handshake status 40'):
                extra_msg = ' (May be due to missing permissions)'
            elif self._websocket_helper.ErrorMsg().startswith('4003'):
                extra_msg = ' (Failed to connect to port %d)' % self._tunnel_target.port
            error_msg = 'Error while connecting [%s].%s' % (self._websocket_helper.ErrorMsg(), extra_msg)
            raise ConnectionCreationError(error_msg)
        raise ConnectionCreationError('Unexpected error while connecting. Check logs for more details.')

    def _OnClose(self):
        self._StopConnectionAsync()

    def _OnData(self, binary_data):
        """Receive a single message from the server."""
        tag, bytes_left = utils.ExtractSubprotocolTag(binary_data)
        if tag == utils.SUBPROTOCOL_TAG_DATA:
            self._HandleSubprotocolData(bytes_left)
        elif tag == utils.SUBPROTOCOL_TAG_ACK:
            self._HandleSubprotocolAck(bytes_left)
        elif tag == utils.SUBPROTOCOL_TAG_CONNECT_SUCCESS_SID:
            self._HandleSubprotocolConnectSuccessSid(bytes_left)
        elif tag == utils.SUBPROTOCOL_TAG_RECONNECT_SUCCESS_ACK:
            self._HandleSubprotocolReconnectSuccessAck(bytes_left)
        else:
            log.debug('Unsupported subprotocol tag [%r], discarding the message', tag)

    def _HandleSubprotocolAck(self, binary_data):
        """Handle Subprotocol ACK Frame."""
        if not self._HasConnected():
            self._StopConnectionAsync()
            raise SubprotocolEarlyAckError('Received ACK before connected.')
        bytes_confirmed, bytes_left = utils.ExtractSubprotocolAck(binary_data)
        self._ConfirmData(bytes_confirmed)
        if bytes_left:
            log.debug('[%d] Discarding [%d] extra bytes after processing ACK', self._conn_id, len(bytes_left))

    def _HandleSubprotocolConnectSuccessSid(self, binary_data):
        """Handle Subprotocol CONNECT_SUCCESS_SID Frame."""
        if self._HasConnected():
            self._StopConnectionAsync()
            raise SubprotocolExtraConnectSuccessSid('Received CONNECT_SUCCESS_SID after already connected.')
        data, bytes_left = utils.ExtractSubprotocolConnectSuccessSid(binary_data)
        self._connection_sid = data
        self._connect_msg_received = True
        if bytes_left:
            log.debug('[%d] Discarding [%d] extra bytes after processing CONNECT_SUCCESS_SID', self._conn_id, len(bytes_left))

    def _AddUnconfirmedDataBackToTheQueue(self):
        for data in self._unconfirmed_data:
            self._data_to_resend.put(data)
        self._unconfirmed_data = collections.deque()

    def _HandleSubprotocolReconnectSuccessAck(self, binary_data):
        """Handle Subprotocol RECONNECT_SUCCESS_ACK Frame."""
        if self._HasConnected():
            self._StopConnectionAsync()
            raise SubprotocolExtraReconnectSuccessAck('Received RECONNECT_SUCCESS_ACK after already connected.')
        bytes_confirmed, bytes_left = utils.ExtractSubprotocolReconnectSuccessAck(binary_data)
        bytes_being_confirmed = bytes_confirmed - self._total_bytes_confirmed
        self._ConfirmData(bytes_confirmed)
        log.info('[%d] Reconnecting: confirming [%d] bytes and resending [%d] messages.', self._conn_id, bytes_being_confirmed, len(self._unconfirmed_data))
        self._AddUnconfirmedDataBackToTheQueue()
        self._connect_msg_received = True
        if bytes_left:
            log.debug('[%d] Discarding [%d] extra bytes after processing RECONNECT_SUCCESS_ACK', self._conn_id, len(bytes_left))

    def _HandleSubprotocolData(self, binary_data):
        """Handle Subprotocol DATA Frame."""
        if not self._HasConnected():
            self._StopConnectionAsync()
            raise SubprotocolEarlyDataError('Received DATA before connected.')
        data, bytes_left = utils.ExtractSubprotocolData(binary_data)
        self._total_bytes_received += len(data)
        self._MaybeSendAck()
        try:
            self._data_handler_callback(data)
        except:
            self._StopConnectionAsync()
            raise
        if bytes_left:
            log.debug('[%d] Discarding [%d] extra bytes after processing DATA', self._conn_id, len(bytes_left))

    def _ConfirmData(self, bytes_confirmed):
        """Discard data that has been confirmed via ACKs received from server."""
        if bytes_confirmed < self._total_bytes_confirmed:
            self._StopConnectionAsync()
            raise SubprotocolOutOfOrderAckError('Received out-of-order Ack for [%d] bytes.' % bytes_confirmed)
        bytes_to_confirm = bytes_confirmed - self._total_bytes_confirmed
        while bytes_to_confirm and self._unconfirmed_data:
            data_chunk = self._unconfirmed_data.popleft()
            if len(data_chunk) > bytes_to_confirm:
                self._unconfirmed_data.appendleft(data_chunk[bytes_to_confirm:])
                self._total_bytes_confirmed += bytes_to_confirm
            else:
                self._total_bytes_confirmed += len(data_chunk)
            bytes_to_confirm = bytes_confirmed - self._total_bytes_confirmed
        if bytes_to_confirm:
            self._StopConnectionAsync()
            raise SubprotocolInvalidAckError('Bytes confirmed [%r] were larger than bytes sent [%r].' % (bytes_confirmed, self._total_bytes_confirmed))