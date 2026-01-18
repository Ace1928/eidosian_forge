import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
class DefaultUrllib3ManagerTest(TestCase):

    def test_no_config(self):
        manager = default_urllib3_manager(config=None)
        self.assertEqual(manager.connection_pool_kw['cert_reqs'], 'CERT_REQUIRED')

    def test_config_no_proxy(self):
        import urllib3
        manager = default_urllib3_manager(config=ConfigDict())
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_config_no_proxy_custom_cls(self):
        import urllib3

        class CustomPoolManager(urllib3.PoolManager):
            pass
        manager = default_urllib3_manager(config=ConfigDict(), pool_manager_cls=CustomPoolManager)
        self.assertIsInstance(manager, CustomPoolManager)

    def test_config_ssl(self):
        config = ConfigDict()
        config.set(b'http', b'sslVerify', b'true')
        manager = default_urllib3_manager(config=config)
        self.assertEqual(manager.connection_pool_kw['cert_reqs'], 'CERT_REQUIRED')

    def test_config_no_ssl(self):
        config = ConfigDict()
        config.set(b'http', b'sslVerify', b'false')
        manager = default_urllib3_manager(config=config)
        self.assertEqual(manager.connection_pool_kw['cert_reqs'], 'CERT_NONE')

    def test_config_proxy(self):
        import urllib3
        config = ConfigDict()
        config.set(b'http', b'proxy', b'http://localhost:3128/')
        manager = default_urllib3_manager(config=config)
        self.assertIsInstance(manager, urllib3.ProxyManager)
        self.assertTrue(hasattr(manager, 'proxy'))
        self.assertEqual(manager.proxy.scheme, 'http')
        self.assertEqual(manager.proxy.host, 'localhost')
        self.assertEqual(manager.proxy.port, 3128)

    def test_environment_proxy(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        manager = default_urllib3_manager(config=config)
        self.assertIsInstance(manager, urllib3.ProxyManager)
        self.assertTrue(hasattr(manager, 'proxy'))
        self.assertEqual(manager.proxy.scheme, 'http')
        self.assertEqual(manager.proxy.host, 'myproxy')
        self.assertEqual(manager.proxy.port, 8080)

    def test_environment_empty_proxy(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', '')
        manager = default_urllib3_manager(config=config)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_1(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh')
        base_url = 'http://xyz.abc.def.gh:8080/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_2(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh,ample.com')
        base_url = 'http://ample.com/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_3(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh,ample.com')
        base_url = 'http://ample.com:80/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_4(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh,ample.com')
        base_url = 'http://www.ample.com/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_5(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh,ample.com')
        base_url = 'http://www.example.com/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertIsInstance(manager, urllib3.ProxyManager)
        self.assertTrue(hasattr(manager, 'proxy'))
        self.assertEqual(manager.proxy.scheme, 'http')
        self.assertEqual(manager.proxy.host, 'myproxy')
        self.assertEqual(manager.proxy.port, 8080)

    def test_environment_no_proxy_6(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,abc.gh,ample.com')
        base_url = 'http://ample.com.org/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertIsInstance(manager, urllib3.ProxyManager)
        self.assertTrue(hasattr(manager, 'proxy'))
        self.assertEqual(manager.proxy.scheme, 'http')
        self.assertEqual(manager.proxy.host, 'myproxy')
        self.assertEqual(manager.proxy.port, 8080)

    def test_environment_no_proxy_ipv4_address_1(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,192.168.0.10,ample.com')
        base_url = 'http://192.168.0.10/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_ipv4_address_2(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,192.168.0.10,ample.com')
        base_url = 'http://192.168.0.10:8888/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_ipv4_address_3(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,ff80:1::/64,192.168.0.0/24,ample.com')
        base_url = 'http://192.168.0.10/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_ipv6_address_1(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,ff80:1::affe,ample.com')
        base_url = 'http://[ff80:1::affe]/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_ipv6_address_2(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,ff80:1::affe,ample.com')
        base_url = 'http://[ff80:1::affe]:1234/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_environment_no_proxy_ipv6_address_3(self):
        import urllib3
        config = ConfigDict()
        self.overrideEnv('http_proxy', 'http://myproxy:8080')
        self.overrideEnv('no_proxy', 'xyz,abc.def.gh,192.168.0.0/24,ff80:1::/64,ample.com')
        base_url = 'http://[ff80:1::affe]/path/port'
        manager = default_urllib3_manager(config=config, base_url=base_url)
        self.assertNotIsInstance(manager, urllib3.ProxyManager)
        self.assertIsInstance(manager, urllib3.PoolManager)

    def test_config_proxy_custom_cls(self):
        import urllib3

        class CustomProxyManager(urllib3.ProxyManager):
            pass
        config = ConfigDict()
        config.set(b'http', b'proxy', b'http://localhost:3128/')
        manager = default_urllib3_manager(config=config, proxy_manager_cls=CustomProxyManager)
        self.assertIsInstance(manager, CustomProxyManager)

    def test_config_proxy_creds(self):
        import urllib3
        config = ConfigDict()
        config.set(b'http', b'proxy', b'http://jelmer:example@localhost:3128/')
        manager = default_urllib3_manager(config=config)
        assert isinstance(manager, urllib3.ProxyManager)
        self.assertEqual(manager.proxy_headers, {'proxy-authorization': 'Basic amVsbWVyOmV4YW1wbGU='})

    def test_config_no_verify_ssl(self):
        manager = default_urllib3_manager(config=None, cert_reqs='CERT_NONE')
        self.assertEqual(manager.connection_pool_kw['cert_reqs'], 'CERT_NONE')