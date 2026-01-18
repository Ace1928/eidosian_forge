import argparse
import os
import sys
import urllib.parse  # noqa: WPS301
from importlib import import_module
from contextlib import suppress
from . import server
from . import wsgi
class GatewayYo:
    """Gateway."""

    def __init__(self, gateway):
        """Init."""
        self.gateway = gateway

    def server(self, parsed_args):
        """Server."""
        server_args = vars(self)
        server_args['bind_addr'] = parsed_args['bind_addr']
        if parsed_args.max is not None:
            server_args['maxthreads'] = parsed_args.max
        if parsed_args.numthreads is not None:
            server_args['minthreads'] = parsed_args.numthreads
        return server.HTTPServer(**server_args)