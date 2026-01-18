from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import image_versions_util as image_version_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import flags
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core.util import semver
def CompareLooseVersions(v1, v2):
    """Compares loose version strings.

  Args:
    v1: first loose version string
    v2: second loose version string

  Returns:
    Value == 1 when v1 is greater; Value == -1 when v2 is greater; otherwise 0.
  """
    v1, v2 = (_VersionStrToLooseVersion(v1), _VersionStrToLooseVersion(v2))
    return _CompareVersions(v1, v2)