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
def _IsVersionUpgradeCompatible(cur_version, candidate_version, image_version_part):
    """Validates whether version candidate is greater than or equal to current.

  Applicable both for Airflow and Composer version upgrades. Composer supports
  both Airflow and self MINOR and PATCH-level upgrades.

  Args:
    cur_version: current 'a.b.c' version
    candidate_version: candidate 'x.y.z' version
    image_version_part: part of image to be validated. Must be either 'Airflow'
      or 'Composer'

  Returns:
    UpgradeValidator namedtuple containing boolean value whether selected image
    version component is valid for upgrade and eventual error message if it is
    not.
  """
    assert image_version_part in ['Airflow', 'Composer']
    curr_semantic_version = _VersionStrToSemanticVersion(cur_version)
    cand_semantic_version = _VersionStrToSemanticVersion(candidate_version)
    if curr_semantic_version > cand_semantic_version:
        error_message = "Upgrade cannot decrease {composer_or_airflow1}'s version. Current {composer_or_airflow2} version: {cur_version}, requested {composer_or_airflow3} version: {req_version}.".format(composer_or_airflow1=image_version_part, composer_or_airflow2=image_version_part, cur_version=cur_version, composer_or_airflow3=image_version_part, req_version=candidate_version)
        return UpgradeValidator(False, error_message)
    if curr_semantic_version.major != cand_semantic_version.major:
        error_message = "Upgrades between different {}'s major versions are not supported. Current major version {}, requested major version {}.".format(image_version_part, curr_semantic_version.major, cand_semantic_version.major)
        return UpgradeValidator(False, error_message)
    return UpgradeValidator(True, None)