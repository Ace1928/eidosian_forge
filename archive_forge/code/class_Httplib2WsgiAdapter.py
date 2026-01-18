import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
class Httplib2WsgiAdapter(object):

    def __init__(self, app):
        self.app = app

    def request(self, uri, method='GET', body=None, headers=None):
        req = webob.Request.blank(uri, method=method, headers=headers)
        if isinstance(body, str):
            req.body = body.encode('utf-8')
        else:
            req.body = body
        resp = req.get_response(self.app)
        return (Httplib2WebobResponse(resp), resp.body.decode('utf-8'))