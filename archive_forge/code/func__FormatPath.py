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
def _FormatPath(self, path):
    path = super(GCSPathExpander, self)._FormatPath(path)
    return 'gs://' if path == 'gs:/' else path