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
def IsVersionComposer3Compatible(image_version):
    """Checks if given `image_version` is compatible with Composer 3.

  Args:
    image_version: image version str that includes Composer version.

  Returns:
    True if Composer version is greater than or equal to 3.0.0 or its prerelease
    variant, otherwise False.
  """
    if image_version:
        version_item = _ImageVersionItem(image_version)
        if version_item and version_item.composer_ver:
            composer_version = version_item.composer_ver
            if composer_version == '3':
                return True
            if composer_version == 'latest':
                composer_version = COMPOSER_LATEST_VERSION_PLACEHOLDER
            return IsVersionInRange(composer_version, flags.MIN_COMPOSER3_VERSION, None, True)
    return False