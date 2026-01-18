import os
import platform
import socket
import tempfile
import testtools
from unittest import mock
import eventlet
import eventlet.wsgi
import requests
import webob
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
from oslo_service import wsgi
from oslo_utils import netutils
class TestLoaderNormalFilesystem(WsgiTestCase):
    """Loader tests with normal filesystem (unmodified os.path module)."""
    _paste_config = '\n[app:test_app]\nuse = egg:Paste#static\ndocument_root = /tmp\n    '

    def setUp(self):
        super(TestLoaderNormalFilesystem, self).setUp()
        self.paste_config = tempfile.NamedTemporaryFile(mode='w+t')
        self.paste_config.write(self._paste_config.lstrip())
        self.paste_config.seek(0)
        self.paste_config.flush()
        self.config(api_paste_config=self.paste_config.name)
        self.loader = wsgi.Loader(CONF)

    def test_config_found(self):
        self.assertEqual(self.paste_config.name, self.loader.config_path)

    def test_app_not_found(self):
        self.assertRaises(wsgi.PasteAppNotFound, self.loader.load_app, 'nonexistent app')

    def test_app_found(self):
        url_parser = self.loader.load_app('test_app')
        self.assertEqual('/tmp', url_parser.directory)

    def tearDown(self):
        self.paste_config.close()
        super(TestLoaderNormalFilesystem, self).tearDown()