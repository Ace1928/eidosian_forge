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
class LocalClient(ClientWrapper):
    """Wraps a dulwich.LocalGitClient."""

    def GetRefs(self):
        with contextlib.closing(repo.Repo(self._path)) as r:
            return r.get_refs()