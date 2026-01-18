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
class UnsupportedClientType(exceptions.Error):
    """Raised when you try to pull from an unknown repository type.

  The URL passed to InstallRuntimeDef identifies an arbitrary git repository.
  This exception is raised when we get one we don't know how to GetRefs() from.
  """