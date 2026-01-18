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
class SSHConnectionPool(urllib3.connectionpool.HTTPConnectionPool):
    scheme = 'ssh'

    def __init__(self, ssh_client=None, timeout=60, maxsize=10, host=None):
        super().__init__('localhost', timeout=timeout, maxsize=maxsize)
        self.ssh_transport = None
        self.timeout = timeout
        if ssh_client:
            self.ssh_transport = ssh_client.get_transport()
        self.ssh_host = host

    def _new_conn(self):
        return SSHConnection(self.ssh_transport, self.timeout, self.ssh_host)

    def _get_conn(self, timeout):
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError:
            raise urllib3.exceptions.ClosedPoolError(self, 'Pool is closed.')
        except queue.Empty:
            if self.block:
                raise urllib3.exceptions.EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.')
        return conn or self._new_conn()