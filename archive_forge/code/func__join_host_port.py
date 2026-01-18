import os
from kubernetes.client import Configuration
from .config_exception import ConfigException
def _join_host_port(host, port):
    """Adapted golang's net.JoinHostPort"""
    template = '%s:%s'
    host_requires_bracketing = ':' in host or '%' in host
    if host_requires_bracketing:
        template = '[%s]:%s'
    return template % (host, port)