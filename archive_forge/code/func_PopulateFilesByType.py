from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
import six.moves.urllib.parse
def PopulateFilesByType(self, args):
    self.files_by_type.update(self.GetFilesByType(args))