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
class TraditionalClient(ClientWrapper):
    """Wraps a dulwich.client.TraditionalGitClient."""

    def GetRefs(self):
        proto = self._transport._connect(b'upload-pack', self._path)[0]
        with proto:
            refs = client.read_pkt_refs(proto)[0]
            proto.write_pkt_line(None)
            return refs