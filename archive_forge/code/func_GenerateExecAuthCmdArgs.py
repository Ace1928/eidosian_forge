from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
def GenerateExecAuthCmdArgs(cluster_id, project_id, location):
    """Returns exec auth provider command args."""
    return ['--use_edge_cloud', '--project', project_id, '--location', location, '--cluster', cluster_id]