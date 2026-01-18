import datetime
import io
import logging
import os
import re
import subprocess
import sys
import time
import unittest
import warnings
import contextlib
import portend
import pytest
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import text_or_bytes, HTTPSConnection, ntob
from cherrypy.lib import httputil
from cherrypy.lib import gctools
@classmethod
def _setup_server(cls, supervisor, conf):
    v = sys.version.split()[0]
    log.info('Python version used to run this test script: %s' % v)
    log.info('CherryPy version: %s' % cherrypy.__version__)
    if supervisor.scheme == 'https':
        ssl = ' (ssl)'
    else:
        ssl = ''
    log.info('HTTP server version: %s%s' % (supervisor.protocol, ssl))
    log.info('PID: %s' % os.getpid())
    cherrypy.server.using_apache = supervisor.using_apache
    cherrypy.server.using_wsgi = supervisor.using_wsgi
    if sys.platform[:4] == 'java':
        cherrypy.config.update({'server.nodelay': False})
    if isinstance(conf, text_or_bytes):
        parser = cherrypy.lib.reprconf.Parser()
        conf = parser.dict_from_file(conf).get('global', {})
    else:
        conf = conf or {}
    baseconf = conf.copy()
    baseconf.update({'server.socket_host': supervisor.host, 'server.socket_port': supervisor.port, 'server.protocol_version': supervisor.protocol, 'environment': 'test_suite'})
    if supervisor.scheme == 'https':
        baseconf['server.ssl_certificate'] = serverpem
        baseconf['server.ssl_private_key'] = serverpem
    if supervisor.scheme == 'https':
        webtest.WebCase.HTTP_CONN = HTTPSConnection
    return baseconf