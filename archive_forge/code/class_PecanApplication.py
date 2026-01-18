from __future__ import print_function
import logging
import os
import sys
import threading
import time
import subprocess
from wsgiref.simple_server import WSGIRequestHandler
from pecan.commands import BaseCommand
from pecan import util
class PecanApplication(WSGIApplication):

    def init(self, parser, opts, args):
        if len(args) != 1:
            parser.error('No configuration file was specified.')
        self.cfgfname = os.path.normpath(os.path.join(os.getcwd(), args[0]))
        self.cfgfname = os.path.abspath(self.cfgfname)
        if not os.path.exists(self.cfgfname):
            parser.error('Config file not found: %s' % self.cfgfname)
        from pecan.configuration import _runtime_conf, set_config
        set_config(self.cfgfname, overwrite=True)
        cfg = {}
        if _runtime_conf.get('server'):
            server = _runtime_conf['server']
            if hasattr(server, 'host') and hasattr(server, 'port'):
                cfg['bind'] = '%s:%s' % (server.host, server.port)
        return cfg

    def load(self):
        from pecan.deploy import deploy
        return deploy(self.cfgfname)