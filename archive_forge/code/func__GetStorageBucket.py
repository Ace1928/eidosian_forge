from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os.path
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _GetStorageBucket(env_ref, release_track=base.ReleaseTrack.GA):
    env = environments_api_util.Get(env_ref, release_track=release_track)
    if not env.config.dagGcsPrefix:
        raise command_util.Error(BUCKET_MISSING_MSG)
    try:
        gcs_dag_dir = storage_util.ObjectReference.FromUrl(env.config.dagGcsPrefix)
    except (storage_util.InvalidObjectNameError, ValueError):
        raise command_util.Error(BUCKET_MISSING_MSG)
    return gcs_dag_dir.bucket_ref