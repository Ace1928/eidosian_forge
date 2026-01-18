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
def _IsComposerMajorOnlyVersionUpgradeCompatible(parsed_curr, parsed_cand):
    """Validates whether Composer major only version upgrade is compatible."""
    if parsed_curr.composer_contains_alias:
        major_version_curr = _GetComposerMajorVersion(parsed_curr.composer_contains_alias[0])
    else:
        major_version_curr = semver.SemVer(parsed_curr.composer_ver).major
    if parsed_cand.composer_contains_alias:
        major_version_cand = _GetComposerMajorVersion(parsed_cand.composer_contains_alias[0])
    else:
        major_version_cand = semver.SemVer(parsed_cand.composer_ver).major
    return UpgradeValidator(major_version_curr == major_version_cand, None)