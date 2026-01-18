from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import base64
import contextlib
import os
import socket
import ssl
import tempfile
import threading
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
@contextlib.contextmanager
def ClusterConnectionInfo(cluster_ref):
    """Get the info we need to use to connect to a GKE cluster.

  Arguments:
    cluster_ref: reference to the cluster to connect to.
  Yields:
    A tuple of (endpoint, ca_certs), where endpoint is the ip address
    of the GKE control plane, and ca_certs is the absolute path of a temporary
    file (lasting the life of the python process) holding the ca_certs to
    connect to the GKE cluster.
  Raises:
    NoCaCertError: if the cluster is missing certificate authority data.
  """
    with calliope_base.WithLegacyQuota():
        adapter = api_adapter.NewAPIAdapter('v1')
        cluster = adapter.GetCluster(cluster_ref)
    auth = cluster.masterAuth
    if auth and auth.clusterCaCertificate:
        ca_data = auth.clusterCaCertificate
    else:
        raise NoCaCertError('Cluster is missing certificate authority data.')
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    files.WriteBinaryFileContents(filename, base64.b64decode(ca_data), private=True)
    try:
        yield (cluster.endpoint, filename)
    finally:
        os.remove(filename)