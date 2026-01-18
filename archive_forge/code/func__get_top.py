from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import string
import subprocess
import sys
import tarfile
import tempfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_creds as v1_creds
from containerregistry.client.v1 import docker_http
import httplib2
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
def _get_top(tarball, name=None):
    """Get the topmost layer in the image tarball."""
    with tarfile.open(name=tarball, mode='r:') as tar:
        reps = tar.extractfile('repositories') or tar.extractfile('./repositories')
        if reps is None:
            raise ValueError('Tarball must contain a repositories file')
        repositories = json.loads(reps.read().decode('utf8'))
    if name:
        key = str(name.as_repository())
        return repositories[key][name.tag]
    if len(repositories) != 1:
        raise ValueError('Tarball must contain a single repository, or a name must be specified to FromTarball.')
    for unused_repo, tags in six.iteritems(repositories):
        if len(tags) != 1:
            raise ValueError('Tarball must contain a single tag, or a name must be specified to FromTarball.')
        for unused_tag, layer_id in six.iteritems(tags):
            return layer_id
    raise Exception('Unreachable code in _get_top()')