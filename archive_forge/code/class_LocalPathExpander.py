from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import fnmatch
import os
import re
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
class LocalPathExpander(PathExpander):
    """Implements path expansion for the local filesystem."""

    def __init__(self):
        super(LocalPathExpander, self).__init__(os.sep)

    def AbsPath(self, path):
        return os.path.abspath(path)

    def IsFile(self, path):
        return os.path.isfile(path)

    def IsDir(self, path):
        return os.path.isdir(path)

    def Exists(self, path):
        return os.path.exists(path)

    def ListDir(self, path):
        try:
            return os.listdir(path)
        except os.error:
            return []

    def Join(self, path1, path2):
        return os.path.join(path1, path2)