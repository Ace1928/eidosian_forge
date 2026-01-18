import paramiko
import queue
import urllib.parse
import requests.adapters
import logging
import os
import signal
import socket
import subprocess
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
import urllib3
import urllib3.connection
class SSHHTTPAdapter(BaseHTTPAdapter):
    __attrs__ = requests.adapters.HTTPAdapter.__attrs__ + ['pools', 'timeout', 'ssh_client', 'ssh_params', 'max_pool_size']

    def __init__(self, base_url, timeout=60, pool_connections=constants.DEFAULT_NUM_POOLS, max_pool_size=constants.DEFAULT_MAX_POOL_SIZE, shell_out=False):
        self.ssh_client = None
        if not shell_out:
            self._create_paramiko_client(base_url)
            self._connect()
        self.ssh_host = base_url
        if base_url.startswith('ssh://'):
            self.ssh_host = base_url[len('ssh://'):]
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.pools = RecentlyUsedContainer(pool_connections, dispose_func=lambda p: p.close())
        super().__init__()

    def _create_paramiko_client(self, base_url):
        logging.getLogger('paramiko').setLevel(logging.WARNING)
        self.ssh_client = paramiko.SSHClient()
        base_url = urllib.parse.urlparse(base_url)
        self.ssh_params = {'hostname': base_url.hostname, 'port': base_url.port, 'username': base_url.username}
        ssh_config_file = os.path.expanduser('~/.ssh/config')
        if os.path.exists(ssh_config_file):
            conf = paramiko.SSHConfig()
            with open(ssh_config_file) as f:
                conf.parse(f)
            host_config = conf.lookup(base_url.hostname)
            if 'proxycommand' in host_config:
                self.ssh_params['sock'] = paramiko.ProxyCommand(host_config['proxycommand'])
            if 'hostname' in host_config:
                self.ssh_params['hostname'] = host_config['hostname']
            if base_url.port is None and 'port' in host_config:
                self.ssh_params['port'] = host_config['port']
            if base_url.username is None and 'user' in host_config:
                self.ssh_params['username'] = host_config['user']
            if 'identityfile' in host_config:
                self.ssh_params['key_filename'] = host_config['identityfile']
        self.ssh_client.load_system_host_keys()
        self.ssh_client.set_missing_host_key_policy(paramiko.RejectPolicy())

    def _connect(self):
        if self.ssh_client:
            self.ssh_client.connect(**self.ssh_params)

    def get_connection(self, url, proxies=None):
        if not self.ssh_client:
            return SSHConnectionPool(ssh_client=self.ssh_client, timeout=self.timeout, maxsize=self.max_pool_size, host=self.ssh_host)
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool:
                return pool
            if self.ssh_client and (not self.ssh_client.get_transport()):
                self._connect()
            pool = SSHConnectionPool(ssh_client=self.ssh_client, timeout=self.timeout, maxsize=self.max_pool_size, host=self.ssh_host)
            self.pools[url] = pool
        return pool

    def close(self):
        super().close()
        if self.ssh_client:
            self.ssh_client.close()