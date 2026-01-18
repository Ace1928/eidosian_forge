import copy
import functools
import socket
import struct
import time
from typing import Any, Optional
import aioquic.quic.configuration  # type: ignore
import aioquic.quic.connection  # type: ignore
import dns.inet
class BaseQuicManager:

    def __init__(self, conf, verify_mode, connection_factory, server_name=None):
        self._connections = {}
        self._connection_factory = connection_factory
        self._session_tickets = {}
        if conf is None:
            verify_path = None
            if isinstance(verify_mode, str):
                verify_path = verify_mode
                verify_mode = True
            conf = aioquic.quic.configuration.QuicConfiguration(alpn_protocols=['doq', 'doq-i03'], verify_mode=verify_mode, server_name=server_name)
            if verify_path is not None:
                conf.load_verify_locations(verify_path)
        self._conf = conf

    def _connect(self, address, port=853, source=None, source_port=0, want_session_ticket=True):
        connection = self._connections.get((address, port))
        if connection is not None:
            return (connection, False)
        conf = self._conf
        if want_session_ticket:
            try:
                session_ticket = self._session_tickets.pop((address, port))
                conf = copy.copy(conf)
                conf.session_ticket = session_ticket
            except KeyError:
                pass
            session_ticket_handler = functools.partial(self.save_session_ticket, address, port)
        else:
            session_ticket_handler = None
        qconn = aioquic.quic.connection.QuicConnection(configuration=conf, session_ticket_handler=session_ticket_handler)
        lladdress = dns.inet.low_level_address_tuple((address, port))
        qconn.connect(lladdress, time.time())
        connection = self._connection_factory(qconn, address, port, source, source_port, self)
        self._connections[address, port] = connection
        return (connection, True)

    def closed(self, address, port):
        try:
            del self._connections[address, port]
        except KeyError:
            pass

    def save_session_ticket(self, address, port, ticket):
        l = len(self._session_tickets)
        if l >= MAX_SESSION_TICKETS:
            keys_to_delete = list(self._session_tickets.keys())[0:SESSIONS_TO_DELETE]
            for key in keys_to_delete:
                del self._session_tickets[key]
        self._session_tickets[address, port] = ticket