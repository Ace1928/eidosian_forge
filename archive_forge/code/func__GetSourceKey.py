from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetSourceKey(obj):
    """Hash the source key of the given RepoSync|RootSync object."""
    source_type = _GetPathValue(obj, ['spec', 'sourceType'])
    if source_type == 'oci':
        return _GetOciKey(obj)
    else:
        return _GetGitKey(obj)