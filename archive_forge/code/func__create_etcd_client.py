import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed.elastic.rendezvous import (
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    """Create a new ``etcd.Client`` from the specified ``RendezvousParameters``."""
    hostname, port = parse_rendezvous_endpoint(params.endpoint, 2379)
    protocol = params.config.get('protocol')
    if protocol is None:
        protocol = 'http'
    elif protocol != 'http' and protocol != 'https':
        raise ValueError('The etcd protocol must be HTTP or HTTPS.')
    ssl_cert = params.config.get('cert')
    if ssl_cert is not None:
        cert_key = params.config.get('key')
        if cert_key is not None:
            ssl_cert = (ssl_cert, cert_key)
    ca_cert = params.config.get('cacert')
    return etcd.Client(hostname, port, protocol=protocol, cert=ssl_cert, ca_cert=ca_cert, allow_reconnect=True)