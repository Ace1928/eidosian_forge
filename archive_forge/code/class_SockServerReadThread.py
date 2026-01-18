import queue
import socket
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from wandb.proto import wandb_server_pb2 as spb
from wandb.sdk.internal.settings_static import SettingsStatic
from ..lib import tracelog
from ..lib.sock_client import SockClient, SockClientClosedError
from .streams import StreamMux
class SockServerReadThread(threading.Thread):
    _sock_client: SockClient
    _mux: StreamMux
    _stopped: 'Event'
    _clients: ClientDict

    def __init__(self, conn: socket.socket, mux: StreamMux, clients: ClientDict) -> None:
        self._mux = mux
        threading.Thread.__init__(self)
        self.name = 'SockSrvRdThr'
        sock_client = SockClient()
        sock_client.set_socket(conn)
        self._sock_client = sock_client
        self._stopped = mux._get_stopped_event()
        self._clients = clients

    def run(self) -> None:
        while not self._stopped.is_set():
            try:
                sreq = self._sock_client.read_server_request()
            except SockClientClosedError:
                break
            assert sreq, 'read_server_request should never timeout'
            sreq_type = sreq.WhichOneof('server_request_type')
            shandler_str = 'server_' + sreq_type
            shandler: Callable[[spb.ServerRequest], None] = getattr(self, shandler_str, None)
            assert shandler, f'unknown handle: {shandler_str}'
            shandler(sreq)

    def stop(self) -> None:
        try:
            self._sock_client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock_client.close()

    def server_inform_init(self, sreq: 'spb.ServerRequest') -> None:
        request = sreq.inform_init
        stream_id = request._info.stream_id
        settings = SettingsStatic(request.settings)
        self._mux.add_stream(stream_id, settings=settings)
        iface = self._mux.get_stream(stream_id).interface
        self._clients.add_client(self._sock_client)
        iface_reader_thread = SockServerInterfaceReaderThread(clients=self._clients, iface=iface, stopped=self._stopped)
        iface_reader_thread.start()

    def server_inform_start(self, sreq: 'spb.ServerRequest') -> None:
        request = sreq.inform_start
        stream_id = request._info.stream_id
        settings = SettingsStatic(request.settings)
        self._mux.update_stream(stream_id, settings=settings)
        self._mux.start_stream(stream_id)

    def server_inform_attach(self, sreq: 'spb.ServerRequest') -> None:
        request = sreq.inform_attach
        stream_id = request._info.stream_id
        self._clients.add_client(self._sock_client)
        inform_attach_response = spb.ServerInformAttachResponse()
        inform_attach_response.settings.CopyFrom(self._mux._streams[stream_id]._settings._proto)
        response = spb.ServerResponse(inform_attach_response=inform_attach_response)
        self._sock_client.send_server_response(response)
        iface = self._mux.get_stream(stream_id).interface
        assert iface

    def server_record_communicate(self, sreq: 'spb.ServerRequest') -> None:
        record = sreq.record_communicate
        record.control.relay_id = self._sock_client._sockid
        stream_id = record._info.stream_id
        iface = self._mux.get_stream(stream_id).interface
        assert iface.record_q
        iface.record_q.put(record)

    def server_record_publish(self, sreq: 'spb.ServerRequest') -> None:
        record = sreq.record_publish
        record.control.relay_id = self._sock_client._sockid
        stream_id = record._info.stream_id
        iface = self._mux.get_stream(stream_id).interface
        assert iface.record_q
        iface.record_q.put(record)

    def server_inform_finish(self, sreq: 'spb.ServerRequest') -> None:
        request = sreq.inform_finish
        stream_id = request._info.stream_id
        self._mux.drop_stream(stream_id)

    def server_inform_teardown(self, sreq: 'spb.ServerRequest') -> None:
        request = sreq.inform_teardown
        exit_code = request.exit_code
        self._mux.teardown(exit_code)