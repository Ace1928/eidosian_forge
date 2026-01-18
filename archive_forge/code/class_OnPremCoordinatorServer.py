import argparse
import json
import logging
import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from ray.autoscaler._private.local.node_provider import LocalNodeProvider
class OnPremCoordinatorServer(threading.Thread):
    """Initializes HTTPServer and serves CoordinatorSenderNodeProvider forever.

    It handles requests from the remote CoordinatorSenderNodeProvider. The
    requests are forwarded to LocalNodeProvider function calls.
    """

    def __init__(self, list_of_node_ips, host, port):
        """Initialize HTTPServer and serve forever by invoking self.run()."""
        logger.info('Running on prem coordinator server on address ' + host + ':' + str(port))
        threading.Thread.__init__(self)
        self._port = port
        self._list_of_node_ips = list_of_node_ips
        address = (host, self._port)
        config = {'list_of_node_ips': list_of_node_ips}
        self._server = HTTPServer(address, runner_handler(LocalNodeProvider(config, cluster_name=None)))
        self.start()

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        """Shutdown the underlying server."""
        self._server.shutdown()
        self._server.server_close()