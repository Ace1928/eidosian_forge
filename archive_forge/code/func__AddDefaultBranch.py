from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as http_exceptions
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files as file_utils
def _AddDefaultBranch(source_archive_url):
    cloud_repo_pattern = '^https://source\\.developers\\.google\\.com/projects/[^/]+/repos/[^/]+$'
    if re.match(cloud_repo_pattern, source_archive_url):
        return source_archive_url + '/moveable-aliases/master'
    return source_archive_url