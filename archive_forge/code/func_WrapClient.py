from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
from dulwich import client
from dulwich import errors
from dulwich import index
from dulwich import porcelain
from dulwich import repo
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
def WrapClient(location):
    """Returns a ClientWrapper."""
    transport, path = client.get_transport_and_path(location)
    if isinstance(transport, client.TraditionalGitClient):
        return TraditionalClient(transport, path)
    elif isinstance(transport, client.HttpGitClient):
        return HTTPClient(transport, path)
    elif isinstance(transport, client.LocalGitClient):
        return LocalClient(transport, path)
    else:
        raise UnsupportedClientType(transport.__class__.__name__)