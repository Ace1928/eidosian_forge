import contextlib
import logging
import random
from socket import IPPROTO_TCP
from socket import TCP_NODELAY
from socket import SHUT_WR
from socket import timeout as SocketTimeout
import ssl
from os_ken import cfg
from os_ken.lib import hub
from os_ken.lib.hub import StreamServer
import os_ken.base.app_manager
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import nx_match
from os_ken.controller import ofp_event
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, DEAD_DISPATCHER
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib import ip
class OpenFlowController(object):

    def __init__(self):
        super(OpenFlowController, self).__init__()
        if not CONF.ofp_tcp_listen_port and (not CONF.ofp_ssl_listen_port):
            self.ofp_tcp_listen_port = ofproto_common.OFP_TCP_PORT
            self.ofp_ssl_listen_port = ofproto_common.OFP_SSL_PORT
            hub.spawn(self.server_loop, ofproto_common.OFP_TCP_PORT_OLD, ofproto_common.OFP_SSL_PORT_OLD)
        else:
            self.ofp_tcp_listen_port = CONF.ofp_tcp_listen_port
            self.ofp_ssl_listen_port = CONF.ofp_ssl_listen_port
        self._clients = {}

    def __call__(self):
        for address in CONF.ofp_switch_address_list:
            addr = tuple(_split_addr(address))
            self.spawn_client_loop(addr)
        self.server_loop(self.ofp_tcp_listen_port, self.ofp_ssl_listen_port)

    def spawn_client_loop(self, addr, interval=None):
        interval = interval or CONF.ofp_switch_connect_interval
        client = hub.StreamClient(addr)
        hub.spawn(client.connect_loop, datapath_connection_factory, interval)
        self._clients[addr] = client

    def stop_client_loop(self, addr):
        client = self._clients.get(addr, None)
        if client is not None:
            client.stop()

    def server_loop(self, ofp_tcp_listen_port, ofp_ssl_listen_port):
        if CONF.ctl_privkey is not None and CONF.ctl_cert is not None:
            p = 'PROTOCOL_TLS'
            ssl_args = {'ssl_ctx': ssl.SSLContext(getattr(ssl, p))}
            ssl_args['ssl_ctx'].options |= ssl.OP_NO_SSLv3 | ssl.OP_NO_SSLv2
            if CONF.ciphers is not None:
                ssl_args['ciphers'] = CONF.ciphers
            if CONF.ca_certs is not None:
                server = StreamServer((CONF.ofp_listen_host, ofp_ssl_listen_port), datapath_connection_factory, keyfile=CONF.ctl_privkey, certfile=CONF.ctl_cert, cert_reqs=ssl.CERT_REQUIRED, ca_certs=CONF.ca_certs, **ssl_args)
            else:
                server = StreamServer((CONF.ofp_listen_host, ofp_ssl_listen_port), datapath_connection_factory, keyfile=CONF.ctl_privkey, certfile=CONF.ctl_cert, **ssl_args)
        else:
            server = StreamServer((CONF.ofp_listen_host, ofp_tcp_listen_port), datapath_connection_factory)
        server.serve_forever()