from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import tarfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
import httplib2
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import six.moves.http_client
def _resolve_tag(self):
    """Resolve the singleton tag this tarball contains using legacy methods."""
    repo_bytes = self._content('repositories', memoize=False)
    repositories = json.loads(repo_bytes.decode('utf8'))
    if len(repositories) != 1:
        raise ValueError('Tarball must contain a single repository, or a name must be specified to FromTarball.')
    for repo, tags in six.iteritems(repositories):
        if len(tags) != 1:
            raise ValueError('Tarball must contain a single tag, or a name must be specified to FromTarball.')
        for tag, unused_layer in six.iteritems(tags):
            return '{repository}:{tag}'.format(repository=repo, tag=tag)
    raise Exception('unreachable')