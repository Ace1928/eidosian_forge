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