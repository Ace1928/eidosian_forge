import json
import os
import posixpath
import stat
import sys
import tempfile
import urllib.parse as urlparse
import zlib
from configparser import ConfigParser
from io import BytesIO
from geventhttpclient import HTTPClient
from ..greenthreads import GreenThreadsMissingObjectFinder
from ..lru_cache import LRUSizeCache
from ..object_store import INFODIR, PACKDIR, PackBasedObjectStore
from ..objects import S_ISGITLINK, Blob, Commit, Tag, Tree
from ..pack import (
from ..protocol import TCP_GIT_PORT
from ..refs import InfoRefsContainer, read_info_refs, write_info_refs
from ..repo import OBJECTDIR, BaseRepo
from ..server import Backend, TCPGitServer
class SwiftConnector:
    """A Connector to swift that manage authentication and errors catching."""

    def __init__(self, root, conf) -> None:
        """Initialize a SwiftConnector.

        Args:
          root: The swift container that will act as Git bare repository
          conf: A ConfigParser Object
        """
        self.conf = conf
        self.auth_ver = self.conf.get('swift', 'auth_ver')
        if self.auth_ver not in ['1', '2']:
            raise NotImplementedError('Wrong authentication version use either 1 or 2')
        self.auth_url = self.conf.get('swift', 'auth_url')
        self.user = self.conf.get('swift', 'username')
        self.password = self.conf.get('swift', 'password')
        self.concurrency = self.conf.getint('swift', 'concurrency') or 10
        self.http_timeout = self.conf.getint('swift', 'http_timeout') or 20
        self.http_pool_length = self.conf.getint('swift', 'http_pool_length') or 10
        self.region_name = self.conf.get('swift', 'region_name') or 'RegionOne'
        self.endpoint_type = self.conf.get('swift', 'endpoint_type') or 'internalURL'
        self.cache_length = self.conf.getint('swift', 'cache_length') or 20
        self.chunk_length = self.conf.getint('swift', 'chunk_length') or 12228
        self.root = root
        block_size = 1024 * 12
        if self.auth_ver == '1':
            self.storage_url, self.token = self.swift_auth_v1()
        else:
            self.storage_url, self.token = self.swift_auth_v2()
        token_header = {'X-Auth-Token': str(self.token)}
        self.httpclient = HTTPClient.from_url(str(self.storage_url), concurrency=self.http_pool_length, block_size=block_size, connection_timeout=self.http_timeout, network_timeout=self.http_timeout, headers=token_header)
        self.base_path = str(posixpath.join(urlparse.urlparse(self.storage_url).path, self.root))

    def swift_auth_v1(self):
        self.user = self.user.replace(';', ':')
        auth_httpclient = HTTPClient.from_url(self.auth_url, connection_timeout=self.http_timeout, network_timeout=self.http_timeout)
        headers = {'X-Auth-User': self.user, 'X-Auth-Key': self.password}
        path = urlparse.urlparse(self.auth_url).path
        ret = auth_httpclient.request('GET', path, headers=headers)
        if ret.status_code < 200 or ret.status_code >= 300:
            raise SwiftException('AUTH v1.0 request failed on ' + '{} with error code {} ({})'.format(str(auth_httpclient.get_base_url()) + path, ret.status_code, str(ret.items())))
        storage_url = ret['X-Storage-Url']
        token = ret['X-Auth-Token']
        return (storage_url, token)

    def swift_auth_v2(self):
        self.tenant, self.user = self.user.split(';')
        auth_dict = {}
        auth_dict['auth'] = {'passwordCredentials': {'username': self.user, 'password': self.password}, 'tenantName': self.tenant}
        auth_json = json.dumps(auth_dict)
        headers = {'Content-Type': 'application/json'}
        auth_httpclient = HTTPClient.from_url(self.auth_url, connection_timeout=self.http_timeout, network_timeout=self.http_timeout)
        path = urlparse.urlparse(self.auth_url).path
        if not path.endswith('tokens'):
            path = posixpath.join(path, 'tokens')
        ret = auth_httpclient.request('POST', path, body=auth_json, headers=headers)
        if ret.status_code < 200 or ret.status_code >= 300:
            raise SwiftException('AUTH v2.0 request failed on ' + '{} with error code {} ({})'.format(str(auth_httpclient.get_base_url()) + path, ret.status_code, str(ret.items())))
        auth_ret_json = json.loads(ret.read())
        token = auth_ret_json['access']['token']['id']
        catalogs = auth_ret_json['access']['serviceCatalog']
        object_store = next((o_store for o_store in catalogs if o_store['type'] == 'object-store'))
        endpoints = object_store['endpoints']
        endpoint = next((endp for endp in endpoints if endp['region'] == self.region_name))
        return (endpoint[self.endpoint_type], token)

    def test_root_exists(self):
        """Check that Swift container exist.

        Returns: True if exist or None it not
        """
        ret = self.httpclient.request('HEAD', self.base_path)
        if ret.status_code == 404:
            return None
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('HEAD request failed with error code %s' % ret.status_code)
        return True

    def create_root(self):
        """Create the Swift container.

        Raises:
          SwiftException: if unable to create
        """
        if not self.test_root_exists():
            ret = self.httpclient.request('PUT', self.base_path)
            if ret.status_code < 200 or ret.status_code > 300:
                raise SwiftException('PUT request failed with error code %s' % ret.status_code)

    def get_container_objects(self):
        """Retrieve objects list in a container.

        Returns: A list of dict that describe objects
                 or None if container does not exist
        """
        qs = '?format=json'
        path = self.base_path + qs
        ret = self.httpclient.request('GET', path)
        if ret.status_code == 404:
            return None
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('GET request failed with error code %s' % ret.status_code)
        content = ret.read()
        return json.loads(content)

    def get_object_stat(self, name):
        """Retrieve object stat.

        Args:
          name: The object name
        Returns:
          A dict that describe the object or None if object does not exist
        """
        path = self.base_path + '/' + name
        ret = self.httpclient.request('HEAD', path)
        if ret.status_code == 404:
            return None
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('HEAD request failed with error code %s' % ret.status_code)
        resp_headers = {}
        for header, value in ret.items():
            resp_headers[header.lower()] = value
        return resp_headers

    def put_object(self, name, content):
        """Put an object.

        Args:
          name: The object name
          content: A file object
        Raises:
          SwiftException: if unable to create
        """
        content.seek(0)
        data = content.read()
        path = self.base_path + '/' + name
        headers = {'Content-Length': str(len(data))}

        def _send():
            ret = self.httpclient.request('PUT', path, body=data, headers=headers)
            return ret
        try:
            ret = _send()
        except Exception:
            ret = _send()
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('PUT request failed with error code %s' % ret.status_code)

    def get_object(self, name, range=None):
        """Retrieve an object.

        Args:
          name: The object name
          range: A string range like "0-10" to
                 retrieve specified bytes in object content
        Returns:
          A file like instance or bytestring if range is specified
        """
        headers = {}
        if range:
            headers['Range'] = 'bytes=%s' % range
        path = self.base_path + '/' + name
        ret = self.httpclient.request('GET', path, headers=headers)
        if ret.status_code == 404:
            return None
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('GET request failed with error code %s' % ret.status_code)
        content = ret.read()
        if range:
            return content
        return BytesIO(content)

    def del_object(self, name):
        """Delete an object.

        Args:
          name: The object name
        Raises:
          SwiftException: if unable to delete
        """
        path = self.base_path + '/' + name
        ret = self.httpclient.request('DELETE', path)
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('DELETE request failed with error code %s' % ret.status_code)

    def del_root(self):
        """Delete the root container by removing container content.

        Raises:
          SwiftException: if unable to delete
        """
        for obj in self.get_container_objects():
            self.del_object(obj['name'])
        ret = self.httpclient.request('DELETE', self.base_path)
        if ret.status_code < 200 or ret.status_code > 300:
            raise SwiftException('DELETE request failed with error code %s' % ret.status_code)