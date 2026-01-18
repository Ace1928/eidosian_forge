from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import os.path
import tarfile
import zipfile
from googlecloudsdk.api_lib.cloudbuild import metric_names
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import files
def _ResetOwnership(tarinfo):
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = 'root'
    return tarinfo