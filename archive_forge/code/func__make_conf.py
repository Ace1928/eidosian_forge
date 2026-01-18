import contextlib
import logging
import os
import shutil
import sys
import tempfile
from urllib import parse as urllib_parse
from taskflow import exceptions
from taskflow.persistence import backends
def _make_conf(backend_uri):
    parsed_url = urllib_parse.urlparse(backend_uri)
    backend_type = parsed_url.scheme.lower()
    if not backend_type:
        raise ValueError('Unknown backend type for uri: %s' % backend_type)
    if backend_type in ('file', 'dir'):
        conf = {'path': parsed_url.path, 'connection': backend_uri}
    elif backend_type in ('zookeeper',):
        conf = {'path': parsed_url.path, 'hosts': parsed_url.netloc, 'connection': backend_uri}
    else:
        conf = {'connection': backend_uri}
    return conf